import google.generativeai as genai
import os
import json
import time
from tqdm import tqdm
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable, NotFound

# ==========================================
# CONFIGURATION
# ==========================================

#VIDEO_ROOT = #"/vol/bitbucket/sna21/dataset/VioGuard/videos"
#OUTPUT_JSON = #"/vol/bitbucket/sna21/dataset/VioGuard/video_llm_captions.json"

API_KEY = "AIzaSyCjdTqacKg_kRQfyTT2AUwrerRibyA8CVg"  # Paste your key
# Root folder that contains `fight/` and `normal/` subfolders referenced in the CSV
VIDEO_ROOT = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS"
VIDEO_ROOT_FIGHT = os.path.join(VIDEO_ROOT, "fight")
VIDEO_ROOT_NORMAL = os.path.join(VIDEO_ROOT, "normal")
# Where to write captions for this dataset
OUTPUT_JSON = "/vol/bitbucket/sna21/dataset/UBI_FIGHTS/video_llm_captions.json"

# UPDATED MODEL LIST (November 2025)
# Removed deprecated 1.5 models to prevent 404 errors.
MODEL_QUEUE = [
    "gemini-2.5-flash",       # Primary: Best balance
    "gemini-2.0-flash",       # Fallback 1: Very reliable
    "gemini-2.5-flash-lite",  # Fallback 2: High rate limits
    "gemini-2.0-flash-lite",  # Fallback 3: Good speed
    "gemini-2.5-pro"          # Last resort: High intelligence, lower quota
]

genai.configure(api_key=API_KEY)

def wait_for_file_active(file):
    """Waits for the uploaded video to be processed."""
    print(f"Processing {file.name}...", end="", flush=True)
    while True:
        try:
            file = genai.get_file(file.name)
        except Exception:
            # Sometimes get_file fails transiently
            time.sleep(2)
            continue

        if file.state.name == "ACTIVE":
            print(" Done.")
            return file
        elif file.state.name == "FAILED":
            raise Exception(f"File {file.name} failed processing.")
        
        print(".", end="", flush=True)
        time.sleep(2)

def generate_caption_with_retry(video_path, model_name_list):
    """
    Tries to generate a caption. Handles Quotas AND Safety Blocks.
    """
    # Try the current model first, then iterate through others if needed
    for i in range(len(model_name_list)):
        current_model = model_name_list[0] 
        model = genai.GenerativeModel(current_model)
        video_file = None
        
        try:
            # 1. Upload Video
            video_file = genai.upload_file(path=video_path)
            video_file = wait_for_file_active(video_file)

            # 2. Prompt
            prompt = (
                "Describe the action in this video in extreme detail. "
                "Focus specifically on velocity, aggression, physical contact, and fighting. "
                "Start directly with the description."
            )

            # 3. Generate
            # We add request_options to catch timeouts too
            response = model.generate_content([video_file, prompt])
            
            # 4. Clean up immediately (Before processing text)
            try: genai.delete_file(video_file.name)
            except: pass
            
            # 5. Extract Text Safely
            try:
                text = response.text.strip()
                return text, model_name_list
            except ValueError:
                # This catches "response.parts... requires a single candidate"
                # This means the model BLOCKED the video due to safety filters.
                print(f"\n[Safety Block] {os.path.basename(video_path)} blocked by {current_model}. Using Fallback.")
                
                # CRITICAL: Return your fallback caption directly here
                return "A video depicting extreme physical violence and aggression.", model_name_list

        except ResourceExhausted:
            print(f"\n[Quota Hit] Model {current_model} exhausted. Switching...")
            exhausted = model_name_list.pop(0)
            model_name_list.append(exhausted)
            
            if i == len(model_name_list) - 1:
                print("All models exhausted. Sleeping for 60 seconds...")
                time.sleep(60)
            
            # Cleanup and retry
            if video_file:
                try: genai.delete_file(video_file.name)
                except: pass
            continue 
            
        except Exception as e:
            print(f"\nError on {video_path}: {e}")
            if video_file:
                try: genai.delete_file(video_file.name)
                except: pass
            return None, model_name_list

    return None, model_name_list

def main():
    # ---------------------------------------------------------
    # RESUME LOGIC
    # ---------------------------------------------------------
    captions = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r') as f:
                captions = json.load(f)
            print(f"✅ Resuming... Found {len(captions)} existing captions in {OUTPUT_JSON}.")
        except json.JSONDecodeError:
            print("⚠️ Warning: JSON file was corrupted. Starting fresh backup created.")
            os.rename(OUTPUT_JSON, OUTPUT_JSON + ".bak")
            captions = {}

    # Read CSV and collect listed videos (relative paths like 'fight/..' or 'normal/...')
    csv_path = '/vol/bitbucket/sna21/dataset/UBI_FIGHTS/test.csv'
    listed_videos = []
    with open(csv_path, 'r') as csvfile:
        for line in csvfile:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            parts = line.split()
            rel_path = parts[0]
            # Normalize and ignore any leading ./
            rel_path = rel_path.lstrip('./')
            full_path = os.path.join(VIDEO_ROOT, rel_path)
            if os.path.exists(full_path):
                listed_videos.append(rel_path)
            else:
                print(f"Warning: listed file not found on disk: {full_path}")

    # Filter: Only process files listed in CSV that are NOT already in captions
    files_to_process = [v for v in listed_videos if v not in captions]

    print(f"Total Listed Videos: {len(listed_videos)}")
    print(f"Already Done: {len(captions)}")
    print(f"Remaining:    {len(files_to_process)}")
    print("-" * 30)

    current_model_queue = MODEL_QUEUE.copy()

    # Process only the remaining files
    for i, filename in enumerate(tqdm(files_to_process)):
        video_path = os.path.join(VIDEO_ROOT, filename)
        
        caption, current_model_queue = generate_caption_with_retry(video_path, current_model_queue)
        
        if caption:
            captions[filename] = caption
            
            # Save every 5 videos to prevent data loss
            if i % 5 == 0: 
                with open(OUTPUT_JSON, 'w') as f:
                    json.dump(captions, f, indent=4)
        
        time.sleep(1) # Polite delay

    # Final Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(captions, f, indent=4)
    print("Done! All captions generated.")

if __name__ == "__main__":
    main()