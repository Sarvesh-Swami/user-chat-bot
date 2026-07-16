import os
import uuid
import time
import shutil
import requests
import torch
import json
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl

# Import your completely untouched original functions from detect_collision.py
from detect_collision import load_detection_model, extract_frames

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Collision Detection Automated Production API",
    description="Unified API server that hosts the model and runs the automated background fetching worker."
)

# Configuration Endpoints for the automated worker loop
FETCH_EVENT_ENDPOINT = "https://pre-abyss-api.dronaaim.ai/ai-model/getVideo"
TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)

# Global variables and explicit device configuration override to bypass editing detect_collision.py
MODEL, PROCESSOR, _ = load_detection_model()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)
MODEL.eval()

# Global state tracking variable to prevent running duplicate workers
WORKER_RUNNING = False

class VideoRequest(BaseModel):
    video_url: HttpUrl

def cleanup_file(path: str):
    """Safely removes temporary files from disk"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")

def run_prediction_pipeline(video_path: str):
    """Core AI processing logic extracting frames and running VideoMAE inference"""
    try:
        logger.info(f"Starting AI prediction pipeline for: {video_path}")
        
        # Extract frames with detailed logging and SmartWitness compatibility
        logger.info("Step 1: Extracting frames from video with SmartWitness compatibility")
        frames = extract_frames(video_path)
        
        if frames is None:
            logger.error("Frame extraction failed - cannot proceed with AI inference")
            logger.info("This may be due to video format issues, corruption, or unsupported codecs")
            return None
        
        logger.info(f"Step 1 completed: Successfully prepared {len(frames)} frames for processing")
        
        # Validate frame consistency
        if len(frames) != 16:
            logger.error(f"Frame count validation failed: expected 16 frames, got {len(frames)}")
            return None
        
        # Process frames through model
        logger.info("Step 2: Preprocessing frames for model input")
        try:
            inputs = PROCESSOR(frames, return_tensors="pt")
            logger.info(f"Preprocessing completed - Input tensor shapes: {[f'{k}: {v.shape}' for k, v in inputs.items()]}")
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {str(e)}")
            logger.error(f"This may indicate incompatible frame formats or memory issues")
            return None
        
        # Move to device
        logger.info(f"Step 3: Moving tensors to device: {DEVICE}")
        try:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logger.info("Device transfer completed successfully")
        except Exception as e:
            logger.error(f"Device transfer failed: {str(e)}")
            logger.error(f"This may indicate GPU memory issues or device compatibility problems")
            return None
        
        # Run inference
        logger.info("Step 4: Running AI model inference")
        try:
            with torch.no_grad():
                outputs = MODEL(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = logits.argmax(-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            logger.info(f"Model inference completed successfully")
            logger.info(f"Raw logits: {logits.cpu().numpy()}")
            logger.info(f"Probabilities: {probabilities.cpu().numpy()}")
            logger.info(f"Predicted class: {predicted_class}, confidence: {confidence:.4f}")
            
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            logger.error(f"This may indicate model compatibility issues or insufficient memory")
            return None
        
        result = {
            "prediction": "collision_detected" if predicted_class == 1 else "no_collision",
            "confidence": round(float(confidence), 4)
        }
        
        logger.info(f"Pipeline completed successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction pipeline: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def continuous_worker_loop():
    """The infinite automated loop that runs continuously on the server thread"""
    global WORKER_RUNNING
    logger.info("Automated Worker Loop has been initiated")
    
    while WORKER_RUNNING:
        logger.info("Requesting next video from event manager queue")
        start_time = time.time()
        
        try:
            # 1. Hit the target endpoint to see if a video event is ready
            logger.info(f"API REQUEST: GET {FETCH_EVENT_ENDPOINT}")
            response = requests.get(FETCH_EVENT_ENDPOINT, timeout=10)
            
            logger.info(f"API RESPONSE: Status={response.status_code}, Headers={dict(response.headers)}")
            
            if response.status_code == 204 or not response.text.strip():
                logger.info(f"No video events available in queue. Response body: '{response.text}'. Sleeping for 10 seconds")
                time.sleep(10)
                continue
                
            response.raise_for_status()
            event_data = response.json()
            logger.info(f"API RESPONSE BODY: {json.dumps(event_data, indent=2)}")
            
            event_id = event_data.get("eventId")
            video_url = event_data.get("media")
            
            if not event_id or not video_url:
                logger.error(f"Malformed queue data received: {event_data}. Skipping")
                time.sleep(2)
                continue
                
            logger.info(f"Event #{event_id}: Found video link. Downloading from: {video_url}")
            
            # 2. Download the video file locally
            temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp4")
            logger.info(f"Event #{event_id}: Downloading video to: {temp_file_path}")
            
            try:
                download_start = time.time()
                with requests.get(str(video_url), stream=True, timeout=30) as r:
                    r.raise_for_status()
                    logger.info(f"Download response: Status={r.status_code}, Content-Length={r.headers.get('content-length', 'unknown')}")
                    
                    with open(temp_file_path, 'wb') as f:
                        bytes_downloaded = 0
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bytes_downloaded += len(chunk)
                
                download_time = time.time() - download_start
                final_size = os.path.getsize(temp_file_path)
                logger.info(f"Event #{event_id}: Download completed in {download_time:.2f}s, file size: {final_size / (1024*1024):.2f} MB")
                
                if final_size == 0:
                    logger.error(f"Event #{event_id}: Downloaded file is empty")
                    cleanup_file(temp_file_path)
                    continue
                    
            except Exception as e:
                logger.error(f"Event #{event_id}: Video download failed: {str(e)}")
                cleanup_file(temp_file_path)
                continue
            
            # 3. Process video directly through the AI pipeline
            logger.info(f"Event #{event_id}: Starting AI processing pipeline")
            result = run_prediction_pipeline(temp_file_path)
            
            # Always clean up the temporary downloaded video file immediately after processing
            cleanup_file(temp_file_path)
            
            if result is None:
                logger.error(f"Event #{event_id}: AI processing pipeline failed - check logs above for specific error")
                continue
                
            execution_time = time.time() - start_time
            logger.info(f"Event #{event_id}: Processing finished in {execution_time:.2f}s")
            logger.info(f"  Prediction: {result['prediction'].upper()}")
            logger.info(f"  Confidence: {result['confidence'] * 100:.2f}%")
            
            UPDATE_SCORE_ENDPOINT = "https://pre-abyss-api.dronaaim.ai/ai-model/updatescore"
            
            # Prepare the payload dynamically using data from this specific video run
            update_payload = {
                "eventId": str(event_id),                               # Sends back the video's eventId
                "status": "collision detected" if result['prediction'] == "collision_detected" else "no collision",
                "score": str(result['confidence'] * 100),              # Convert score to a string percentage matching your curl format
                "exec_time": f"{execution_time:.2f}"                    # Formatted execution time string
            }
            
            logger.info(f"Event #{event_id}: Sending prediction updates back to backend")
            logger.info(f"API REQUEST: POST {UPDATE_SCORE_ENDPOINT}")
            logger.info(f"REQUEST PAYLOAD: {json.dumps(update_payload, indent=2)}")
            
            try:
                update_response = requests.post(UPDATE_SCORE_ENDPOINT, json=update_payload, timeout=10)
                
                logger.info(f"UPDATE API RESPONSE: Status={update_response.status_code}, Headers={dict(update_response.headers)}")
                
                update_response.raise_for_status()
                
                # Try to log response body if it exists
                response_text = update_response.text.strip()
                if response_text:
                    try:
                        response_json = update_response.json()
                        logger.info(f"UPDATE API RESPONSE BODY: {json.dumps(response_json, indent=2)}")
                    except:
                        logger.info(f"UPDATE API RESPONSE BODY: {response_text}")
                else:
                    logger.info("UPDATE API RESPONSE BODY: (empty)")
                
                logger.info(f"Event #{event_id}: Backend updated successfully! Status: {update_response.status_code}")
                
            except Exception as e:
                logger.error(f"Failed to update backend for Event #{event_id}: {e}")
            # ===================================================================
            # NOTE: If you need to send these results back to your database api, 
            # you can add a simple requests.post() line here using event_id and result data.

            # Loop cycles back instantly to pull the next available video without sleeping
            
        except Exception as e:
            logger.error(f"Worker Loop encountered an error: {e}")
            logger.info("Backing off for 15 seconds before retrying")
            time.sleep(15)


# ================= API ENDPOINTS =================

@app.post("/predict")
async def predict_single_url(payload: VideoRequest, background_tasks: BackgroundTasks):
    """Manual Endpoint: Accepts an individual payload url on demand"""
    start_time = time.time()
    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp4")
    
    try:
        logger.info(f"Manual prediction request for URL: {payload.video_url}")
        
        # Download video
        logger.info("Downloading video from provided URL")
        download_start = time.time()
        with requests.get(str(payload.video_url), stream=True, timeout=30) as r:
            r.raise_for_status()
            logger.info(f"Download response: Status={r.status_code}, Content-Length={r.headers.get('content-length', 'unknown')}")
            
            with open(temp_file_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
        download_time = time.time() - download_start
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Video downloaded in {download_time:.2f}s, file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size == 0:
            raise HTTPException(status_code=422, detail="Downloaded video file is empty")
                
        background_tasks.add_task(cleanup_file, temp_file_path)
        
        # Process video
        logger.info("Starting AI processing pipeline")
        result = run_prediction_pipeline(temp_file_path)
        
        if result is None:
            raise HTTPException(status_code=422, detail="Failed to process video - check server logs for details")
            
        execution_time = time.time() - start_time
        logger.info(f"Manual prediction completed in {execution_time:.2f}s")
        
        return {
            "status": "success",
            **result,
            "execution_time_seconds": round(execution_time, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/start-worker")
async def start_automated_worker(background_tasks: BackgroundTasks):
    """Trigger Endpoint: Automatically wakes up the permanent loop worker in the background"""
    global WORKER_RUNNING
    if WORKER_RUNNING:
        return {"status": "ignored", "message": "Automated background fetching worker is already running."}
        
    WORKER_RUNNING = True
    background_tasks.add_task(continuous_worker_loop)
    return {"status": "success", "message": "Automated pipeline worker started successfully."}

@app.post("/stop-worker")
async def stop_automated_worker():
    """Management Endpoint: Safely halts the automated loop sequence gracefully"""
    global WORKER_RUNNING
    if not WORKER_RUNNING:
        return {"status": "ignored", "message": "Worker loop is not running currently."}
        
    WORKER_RUNNING = False
    return {"status": "success", "message": "Stop signal sent. The loop will halt as soon as the current event finishes."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)