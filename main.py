from flask import Flask, jsonify, request
import os
import json
import cv2
import numpy as np
from pathlib import Path
import tempfile
import uuid
from supabase import create_client, Client
import io
from PIL import Image
from dotenv import load_dotenv
import gc

# Load environment variables from .env.local file if it exists
load_dotenv('.env.local')

app = Flask(__name__)

# Constants
BUCKET_NAME = 'images'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8m.pt')

# Global model instance (loaded once)
_model = None

def get_model():
    """Get or create the YOLO model instance"""
    global _model
    if _model is None:
        try:
            # Import ultralytics only when needed to avoid startup issues
            from ultralytics import YOLO
            _model = YOLO(MODEL_PATH)
        except Exception as e:
            print("⚠️ ultralytics not available")
            return None
    return _model

def create_supabase_client():
    """Create and return a Supabase client"""
    print("Inside create_supabase_client function")
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    supabase_access_key_id = os.environ.get('SUPABASE_ACCESS_KEY_ID')
    supabase_secret_access_key = os.environ.get('SUPABASE_SECRET_ACCESS_KEY')
    
    print(f"URL exists: {bool(supabase_url)}")
    print(f"Service role key exists: {bool(supabase_service_role_key)}")
    print(f"Access key ID exists: {bool(supabase_access_key_id)}")
    print(f"Secret access key exists: {bool(supabase_secret_access_key)}")
    
    if not supabase_url:
        raise ValueError('SUPABASE_URL not configured')
    
    # Use service role key for full access to private storage
    if supabase_service_role_key:
        print("Using service role key authentication")
        return create_client(supabase_url, supabase_service_role_key)
    elif supabase_access_key_id and supabase_secret_access_key:
        print("Using access key authentication")
        # For now, let's try creating a basic client without the complex auth
        # The access keys might be handled differently
        return create_client(supabase_url, supabase_access_key_id)
    else:
        raise ValueError('Supabase authentication credentials not configured. Need either SUPABASE_SERVICE_ROLE_KEY or ACCESS_KEY_ID/SECRET_ACCESS_KEY')

@app.route('/')
def index():
    return jsonify({"Hello": "World"})

@app.route('/admin/books/crop', methods=['POST'])
def crop_books():
    try:
        # Parse request body
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        body = request.get_json()
        file_name = body.get('fileName')
        
        if not file_name:
            return jsonify({'error': 'fileName is required'}), 400

        print(f'Processing book detection for: {file_name}')

        # Create Supabase client
        try:
            print("Creating Supabase client...")
            supabase = create_supabase_client()
            print("Supabase client created successfully")
        except Exception as e:
            print(f"Supabase client creation error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

        # Download the image from Supabase
        try:
            print(f"Downloading image: {file_name} from bucket: {BUCKET_NAME}")
            response = supabase.storage.from_(BUCKET_NAME).download(file_name)
            print(f"Download response type: {type(response)}")
            image_data = response
            print("Image downloaded successfully")
        except Exception as e:
            print(f'Supabase download error: {e}')
            print(f'Error type: {type(e)}')
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Failed to download image from storage'}), 500

        # Convert image data to OpenCV format
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 500

        # Run YOLO inference using global model
        model = get_model()
        if model is None:
            return jsonify({'error': 'YOLO model not available'}), 500
            
        results = model(image)

        # Process results and crop books
        summary_image = image.copy()
        cropped_book_urls = []
        book_count = 0

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])

                if label == "book" and confidence > 0.3:  # Only process books with confidence > 0.3
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop the book
                    cropped = image[y1:y2, x1:x2]
                    
                    # Convert cropped image to bytes
                    _, buffer = cv2.imencode('.jpg', cropped)
                    cropped_bytes = buffer.tobytes()
                    
                    # Upload cropped book to Supabase
                    book_file_name = f"book-cropped-{uuid.uuid4()}.jpg"
                    
                    try:
                        supabase.storage.from_(BUCKET_NAME).upload(
                            book_file_name, 
                            cropped_bytes,
                            file_options={
                                'content-type': 'image/jpeg',
                                'cache-control': '3600',
                                'upsert': False
                            }
                        )
                        
                        # Generate signed URL
                        signed_url_response = supabase.storage.from_(BUCKET_NAME).create_signed_url(
                            book_file_name, 
                            60 * 60 * 24 * 7  # 7 days expiry
                        )
                        
                        if signed_url_response:
                            cropped_book_urls.append(signed_url_response['signedURL'])
                            
                    except Exception as e:
                        print(f'Failed to upload cropped book: {e}')
                    
                    # Draw bounding box on summary image
                    color = (0, 255, 0)  # Green
                    thickness = 3
                    cv2.rectangle(summary_image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label
                    label_text = f"book_{book_count} ({confidence:.2f})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_color = (0, 255, 255)  # Yellow
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, font, font_scale, font_thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(summary_image, 
                                (x1, y1 - text_height - 10), 
                                (x1 + text_width + 10, y1), 
                                (0, 0, 0), -1)  # Black background
                    
                    # Draw text
                    cv2.putText(summary_image, label_text, (x1 + 5, y1 - 5), 
                              font, font_scale, text_color, font_thickness)
                    
                    book_count += 1

        print(f'Found {book_count} books with confidence > 0.3')

        # Create and upload summary image (optional - commented out for now)
        summary_url = ""
        # if book_count > 0:
        #     _, summary_buffer = cv2.imencode('.jpg', summary_image)
        #     summary_bytes = summary_buffer.tobytes()
        #     summary_file_name = f"book-summary-{uuid.uuid4()}.jpg"
        #     
        #     try:
        #         supabase.storage.from_(BUCKET_NAME).upload(
        #             summary_file_name, 
        #             summary_bytes,
        #             file_options={'content-type': 'image/jpeg', 'cache-control': '3600'}
        #         )
        #         
        #         summary_signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(
        #             summary_file_name, 60 * 60 * 24 * 7
        #         )
        #         if summary_signed_url:
        #             summary_url = summary_signed_url['signedURL']
        #     except Exception as e:
        #         print(f'Failed to upload summary image: {e}')

        # Send successful response
        response_data = {
            'success': True,
            'data': {
                'totalBooks': book_count,
                'croppedBookUrls': cropped_book_urls,
                'summaryUrl': summary_url,
                'originalFileName': file_name
            }
        }
        
        # Memory cleanup
        del image, summary_image, image_array, results
        if 'cropped' in locals():
            del cropped
        if 'cropped_bytes' in locals():
            del cropped_bytes
        gc.collect()
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f'Error in book crop processing: {e}')
        # Clean up memory on error too
        if 'image' in locals():
            del image
        if 'image_array' in locals():
            del image_array
        gc.collect()
        return jsonify({'error': 'Failed to process book cropping. Please try again.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
