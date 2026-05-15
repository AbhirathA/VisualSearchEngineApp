import streamlit as st
import torch
import os

# Fix for "RuntimeError: could not create a primitive" on Windows CPU
torch.backends.mkldnn.enabled = False
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
import torch.nn.functional as F
from ultralytics import YOLO
import faiss
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import json
from streamlit_cropper import st_cropper

# --- Page Config & Styling ---
st.set_page_config(page_title="Visual Search Engine", layout="wide", page_icon="🔍")

st.markdown("""
<style>
    .result-caption {
        font-size: 0.9em;
        color: #ddd;
        background-color: #222;
        padding: 10px;
        border-radius: 5px;
        margin-top: 5px;
        min-height: 80px;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching Data & Models ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolo_finetune_outputs/best.pt')

@st.cache_resource
def load_clip_model():
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    # Load finetuned weights
    try:
        model.load_state_dict(torch.load('clip_c_output/full_clip_finetuned.pt', map_location='cpu'))
    except Exception as e:
        st.error(f"Error loading fine-tuned CLIP weights: {e}")
    model.eval()
    return processor, model

@st.cache_resource
def load_faiss_index():
    return faiss.read_index('clip_c_output/idx_C_b07.bin')

@st.cache_resource
def load_blip_itm_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    model.eval()
    return processor, model

@st.cache_resource
def load_blip_caption_model():
    """BLIP image captioner — used to generate a text description of the query crop."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

@st.cache_data
def load_metadata():
    try:
        item_map = pd.read_csv('clip_c_output/item_index_map.csv')
    except FileNotFoundError:
        st.error("❌ 'item_index_map.csv' not found in 'clip_c_output/'. Please ensure it exists.")
        item_map = pd.DataFrame()
        
    try:
        with open('Blip_captions_output/gallery_captions.json', 'r') as f:
            captions = json.load(f)
    except FileNotFoundError:
        st.error("❌ 'gallery_captions.json' not found in 'Blip_captions_output/'.")
        captions = {}
        
    return item_map, captions

# Initialize models
with st.spinner("Loading models... This may take a minute on the first run."):
    yolo_model = load_yolo_model()
    clip_processor, clip_model = load_clip_model()
    blip_itm_processor, blip_itm_model = load_blip_itm_model()
    blip_cap_processor, blip_cap_model = load_blip_caption_model()
    faiss_index = load_faiss_index()
    item_map, gallery_captions = load_metadata()

# --- Main App UI ---
st.markdown("<h1 class='main-header'>🛍️ AI Fashion Visual Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to automatically detect clothing items and find visually similar products!</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display uploaded image
    original_image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("1. Object Detection")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
        
    # 2. Run YOLO
    with st.spinner("Detecting clothing items..."):
        results = yolo_model(original_image)[0]
    
    crops = []
    crop_labels = []
    
    # Process YOLO boxes if any exist
    if len(results.boxes) > 0:
        img_with_boxes = original_image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = yolo_model.names[cls_id]
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{cls_name} ({conf:.2f})", fill="red")
            
            crop = original_image.crop((x1, y1, x2, y2))
            crops.append(crop)
            crop_labels.append(f"{cls_name} (Conf: {conf:.2f})")
            
        with col2:
            st.image(img_with_boxes, caption="Detected Items", use_container_width=True)
    else:
        st.warning("No clothing items detected by YOLO. You can manually crop the image below.")
            
    st.divider()
        
    # Add Manual Crop option
    crop_labels.append("Re-crop (Manual Correction)")
    crops.append(None)
    
    # 3. Item Selection
    st.subheader("2. Select Item to Search")
    
    selected_crop_idx = st.radio(
        "Choose which detected item you want to find matches for:",
        range(len(crops)),
        format_func=lambda x: crop_labels[x],
        horizontal=True
    )
    
    # Show the selected crop or the cropper
    if crops[selected_crop_idx] is None:
        st.info("Drag the corners of the red box over the image to crop the exact item you want to search for.")
        selected_crop = st_cropper(original_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        st.image(selected_crop, width=200, caption="Your Manual Crop Preview")
    else:
        selected_crop = crops[selected_crop_idx]
        st.image(selected_crop, width=200, caption=f"Selected: {crop_labels[selected_crop_idx]}")
        
    st.divider()
    # 4. Feature Extraction & FAISS Search
    k_input = st.slider("Number of top results to retrieve (k)", min_value=1, max_value=50, value=12)

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    run_standard = btn_col1.button("🔍 Find Similar Items", type="secondary", use_container_width=True)
    run_fast     = btn_col2.button("⚡ Fast Search", type="primary", use_container_width=True)
    run_deep     = btn_col3.button("🧠 Deep Search", type="primary", use_container_width=True)

    if run_standard or run_fast or run_deep:
        if item_map.empty:
            st.error("Cannot perform search: item_index_map.csv is missing.")
        else:
            st.subheader("3. Search Results")
            with st.spinner("Extracting features and searching gallery..."):
                # ── CLIP image embedding (shared by all three modes) ──────────
                img_inputs = clip_processor(images=selected_crop, return_tensors="pt")
                with torch.no_grad():
                    raw = clip_model.get_image_features(**img_inputs)
                    image_features = raw.pooler_output if hasattr(raw, 'pooler_output') and not isinstance(raw, torch.Tensor) else raw
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                if run_deep:
                    # ── Deep Search: fuse image + caption embeddings ──────────
                    # Step 1: Generate a caption for the query crop via BLIP
                    cap_inputs = blip_cap_processor(images=selected_crop, return_tensors="pt")
                    with torch.no_grad():
                        cap_ids = blip_cap_model.generate(**cap_inputs, max_new_tokens=50)
                    query_caption = blip_cap_processor.decode(cap_ids[0], skip_special_tokens=True)
                    st.info(f"🖊️ Generated caption: *{query_caption}*")

                    # Step 2: CLIP text embedding of the generated caption
                    txt_inputs = clip_processor(text=[query_caption], return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        text_features = clip_model.get_text_features(**txt_inputs)
                        text_features = text_features.pooler_output if hasattr(text_features, 'pooler_output') and not isinstance(text_features, torch.Tensor) else text_features
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Step 3: Fuse — v = α·φ_V + (1−α)·φ_T, then normalize
                    alpha = 0.7
                    fused = alpha * image_features + (1 - alpha) * text_features
                    fused = fused / fused.norm(dim=-1, keepdim=True)
                    query_vec = fused.numpy()
                else:
                    query_vec = image_features.numpy()

                # Search FAISS
                k = k_input
                distances, indices = faiss_index.search(query_vec, k)

                # Build candidate list
                candidates = []
                for dist, idx in zip(distances[0], indices[0]):
                    try:
                        row = item_map[item_map['faiss_index_pos'] == idx].iloc[0]
                        candidates.append({
                            'dist': dist,
                            'idx': idx,
                            'item_id': row['item_id'],
                            'image_name': row['image_name'],
                            'caption': gallery_captions.get(row['image_name'], "No caption available.")
                        })
                    except IndexError:
                        continue

            # ── Re-ranking: choose method based on which button was clicked ──
            if run_standard and not run_deep:
                # Original method: one full BLIP forward pass per candidate
                with st.spinner("Re-ranking candidates using BLIP ITM (standard)..."):
                    for cand in candidates:
                        if cand['caption'] == "No caption available.":
                            cand['itm_score'] = 0.0
                            continue

                        blip_inputs = blip_itm_processor(
                            images=selected_crop, text=cand['caption'], return_tensors="pt"
                        )
                        with torch.no_grad():
                            itm_output = blip_itm_model(**blip_inputs)
                            itm_score = F.softmax(itm_output.itm_score, dim=1)[0][1].item()
                        cand['itm_score'] = itm_score

            else:
                # Fast method: encode image once, single batched pass for all captions
                with st.spinner("Re-ranking candidates using BLIP ITM (fast)..."):
                    # Step 1: Encode the query image ONCE
                    pixel_values = blip_itm_processor(
                        images=selected_crop, return_tensors="pt"
                    ).pixel_values
                    with torch.no_grad():
                        vision_outputs = blip_itm_model.vision_model(pixel_values)
                        image_embeds = vision_outputs.last_hidden_state   # (1, seq_len, hidden)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

                    # Step 2: Assign 0 to no-caption candidates
                    scored_cands = [c for c in candidates if c['caption'] != "No caption available."]
                    for c in candidates:
                        if c['caption'] == "No caption available.":
                            c['itm_score'] = 0.0

                    # Step 3: Single batched forward pass for all captions
                    if scored_cands:
                        captions = [c['caption'] for c in scored_cands]
                        text_inputs = blip_itm_processor(
                            text=captions,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                        batch_size = len(captions)

                        with torch.no_grad():
                            expanded_image_embeds = image_embeds.expand(batch_size, -1, -1)
                            expanded_image_atts   = image_atts.expand(batch_size, -1)

                            text_outputs = blip_itm_model.text_encoder(
                                input_ids=text_inputs.input_ids,
                                attention_mask=text_inputs.attention_mask,
                                encoder_hidden_states=expanded_image_embeds,
                                encoder_attention_mask=expanded_image_atts,
                                return_dict=True,
                            )

                            itm_logits = blip_itm_model.itm_head(
                                text_outputs.last_hidden_state[:, 0, :]
                            )
                            itm_probs = F.softmax(itm_logits, dim=1)[:, 1].tolist()

                        for cand, score in zip(scored_cands, itm_probs):
                            cand['itm_score'] = score

            # Sort by ITM score (both paths)
            candidates.sort(key=lambda x: x['itm_score'], reverse=True)

            # ── Display Results in Grid ───────────────────────────────────────
            cols = st.columns(4)

            for i, cand in enumerate(candidates):
                col = cols[i % 4]

                dist       = cand['dist']
                item_id    = cand['item_id']
                image_name = cand['image_name']
                caption    = cand['caption']
                itm_score  = cand['itm_score']

                # Determine the correct local path for the image crop
                if image_name.startswith('img/'):
                    relative_path = image_name[4:]   # strip "img/"
                else:
                    relative_path = image_name

                crop_path = os.path.join("yolo_crops", "data", "bbox_crops", relative_path)

                if not os.path.exists(crop_path):
                    crop_path = None

                with col:
                    with st.container(border=True):
                        if crop_path and os.path.exists(crop_path):
                            img = Image.open(crop_path)
                            img = ImageOps.pad(img, (300, 400), color="white")
                            st.image(img, use_container_width=True)
                        else:
                            st.warning("Image missing")

                        st.markdown(f"**Similarity (CLIP):** {dist:.4f}")
                        st.markdown(f"**ITM Score (BLIP):** {itm_score:.4f}")
                        st.markdown(f"**Item ID:** {item_id}")
                        st.markdown(f"<div class='result-caption'>{caption}</div>", unsafe_allow_html=True)
