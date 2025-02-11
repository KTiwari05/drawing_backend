from flask import Flask, request, jsonify, make_response, send_file, abort
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from datetime import timedelta
from models import db, User
from PIL import Image
import fitz
import io
import os
import tempfile
from werkzeug.utils import safe_join
import cv2
import numpy as np
import uuid
 
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])
 
# Configuration settings
app.config.from_object("config.ApplicationConfig")
app.config['JWT_SECRET_KEY'] = app.config['SECRET_KEY']
app.config['JWT_COOKIE_SECURE'] = False
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_ACCESS_COOKIE_NAME'] = 'access_token_cookie'  # Ensure consistency
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Disable CSRF protection
 
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB file limit
 
# Disable CSRF protection for API routes (if using Flask-WTF or Flask-SeaSurf)
app.config['WTF_CSRF_ENABLED'] = False
 
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
db.init_app(app)
 
with app.app_context():
    db.create_all()
 
# User registration route
@app.route("/register", methods=["POST"])
def register_user():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"message": "Email and password are required"}), 400
 
    user_exists = User.query.filter_by(email=data["email"]).first() is not None
    if user_exists:
        return jsonify({"message": "User with that email already exists"}), 409
 
    hashed_password = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    new_user = User(email=data["email"], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
 
    return jsonify({"id": new_user.id, "email": new_user.email}), 201
 
# User login route
@app.route("/login", methods=["POST"])
def login_user():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"message": "Email and password are required"}), 400
 
    user = User.query.filter_by(email=data["email"]).first()
    if user is None or not bcrypt.check_password_hash(user.password, data["password"]):
        return jsonify({"message": "Invalid email or password"}), 401
 
    access_token = create_access_token(identity=user.id, expires_delta=timedelta(hours=1/2))
    response = make_response(jsonify({
        "message": "Login successful",
        "id": user.id,
        "email": user.email,
        "token": access_token
    }))
    response.set_cookie(
        'access_token_cookie',
        access_token,
        httponly=True,
        samesite='Lax',
        path='/'
    )
 
    return response, 200
 
# User logout route
@app.route("/logout", methods=["POST"])
@jwt_required()
def logout_user():
    response = make_response(jsonify({"message": "Logout successful"}))
    response.delete_cookie('access_token_cookie', path='/')
    return response, 200
 
# Get current user route
@app.route("/@me", methods=["GET"])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({"message": "User not found"}), 404
 
    return jsonify({"id": user.id, "email": user.email}), 200
 
# PDF to image conversion function

def pdf_to_png(file):
    images = []
    file_ext = file.name.lower().split('.')[-1]  # Get file extension
    
    if file_ext == "pdf":
        # Handle PDF conversion
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page_no in range(len(pdf)):
            pix = pdf[page_no].get_pixmap(dpi=200)  # High DPI for quality
            if pix.width * pix.height * 3 > 178956970:  # If too large, reduce DPI
                pix = pdf[page_no].get_pixmap(dpi=150)
                if pix.width * pix.height * 3 > 178956970:  # If too large, reduce DPI
                    pix = pdf[page_no].get_pixmap(dpi=100)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(image)
        pdf.close()
    
    elif file_ext in ["tif", "tiff"]:
        # Handle TIFF conversion (multi-page TIFF support)
        tiff_image = Image.open(file)
        try:
            for frame in range(tiff_image.n_frames):  # Extract all pages
                tiff_image.seek(frame)
                images.append(tiff_image.copy())  # Copy current frame to list
        except EOFError:
            pass  # No more frames
    
    return images

def align_images(image1, image2):
    """Aligns two images using ORB feature matching and homography."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

    sift = cv2.SIFT_create()  # SIFT detector (better accuracy)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Feature matching using FLANN based matcher for faster performance
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Use the good matches for homography
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            aligned_img1 = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]), flags=cv2.INTER_LINEAR)
        else:
            aligned_img1 = image1  # Return original if homography fails
    else:
        aligned_img1 = image1  # Return original if not enough matches

    return aligned_img1, image2  # Return aligned image1 and unchanged image2

def resize_and_pad(image1, image2):
    """Resize and pad images to have the same dimensions."""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    aspect_ratio1 = w1 / h1
    aspect_ratio2 = w2 / h2

    if abs(aspect_ratio1 - aspect_ratio2) < 0.05:  # Same aspect ratio, just resize
        target_size = (max(w1, w2), max(h1, h2))
        image1_resized = cv2.resize(image1, target_size)
        image2_resized = cv2.resize(image2, target_size)
    else:  # Different aspect ratios, use padding
        max_width = max(w1, w2)
        max_height = max(h1, h2)
        image1_resized = np.full((max_height, max_width, 3), 255, dtype=np.uint8)
        image2_resized = np.full((max_height, max_width, 3), 255, dtype=np.uint8)

        # Center the images
        y_offset1 = (max_height - h1) // 2
        x_offset1 = (max_width - w1) // 2
        y_offset2 = (max_height - h2) // 2
        x_offset2 = (max_width - w2) // 2

        image1_resized[y_offset1:y_offset1+h1, x_offset1:x_offset1+w1] = image1
        image2_resized[y_offset2:y_offset2+h2, x_offset2:x_offset2+w2] = image2

    return image1_resized, image2_resized

def highlight_differences_separately(image1, image2):

    # Align images
    aligned_image1, aligned_image2 = align_images(image1, image2)
    aligned_image1, aligned_image2 = resize_and_pad(aligned_image1, aligned_image2)

    # Convert to grayscale
    gray_image1 = cv2.cvtColor(aligned_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(aligned_image2, cv2.COLOR_BGR2GRAY)

    diff1 = cv2.subtract(gray_image1, gray_image2)
    diff2 = cv2.subtract(gray_image2, gray_image1)
    _, threshold_diff1 = cv2.threshold(diff1, 50, 255, cv2.THRESH_BINARY)
    _, threshold_diff2 = cv2.threshold(diff2, 50, 255, cv2.THRESH_BINARY)
    # contours_old, _ = cv2.findContours(threshold_diff1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_new, _ = cv2.findContours(threshold_diff2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # highlighted = cv2.cvtColor(gray_image2, cv2.COLOR_GRAY2BGR)
    # for contour in contours_old:
    #     if cv2.contourArea(contour) > 10:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # for contour in contours_new:
    #     if cv2.contourArea(contour) > 10:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # return highlighted
    # Reduce background intensity for better visibility
    highlighted_differences = cv2.cvtColor(gray_image2, cv2.COLOR_GRAY2BGR)
    # highlighted_differences = (highlighted_differences * 0.9).astype(np.uint8)  # Darken background
    highlighted_differences = cv2.addWeighted(highlighted_differences, 1, np.zeros_like(highlighted_differences), 0, 2)  # Lighten background
    highlighted_differences[threshold_diff1>0] = [50, 255, 200]
    
    highlighted_differences[threshold_diff2>0] = [100, 0, 200]
    # highlighted_differences = cv2.addWeighted(highlighted_differences, 0.5, mask_green, 0.7, 0)  # Green overlay
    # highlighted_differences = cv2.addWeighted(highlighted_differences, 0.7, mask_red, 1, 1)  # Red overlay (priority)

    return highlighted_differences

def save_images_as_pdf(images, output_pdf_path):
    rgb_images = [img.convert('RGB') for img in images]
    rgb_images[0].save(output_pdf_path, save_all=True, append_images=rgb_images[1:], resolution=100.0)


# PDF comparison route
@app.route('/compare', methods=['POST'])
@jwt_required()
def compare_files():
    
    old_path = None
    new_path = None
    try:
        user_id = get_jwt_identity()
        print(f"Authenticated user ID: {user_id}")
 
        if 'file1' not in request.files or 'file2' not in request.files:
            print("Missing files in the request.")
            return jsonify({"error": "Both files are required for comparison."}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        if not file1 or not file2:
            print("One or both files are empty.")
            return jsonify({"error": "Both files must be uploaded."}), 400

        # Generate unique filenames in the static folder
        old_path = os.path.join('static', f"{uuid.uuid4()}.pdf")
        new_path = os.path.join('static', f"{uuid.uuid4()}.pdf")

        # Save uploaded files
        file1.save(old_path)
        file2.save(new_path)
        print(f"Files saved to: {old_path}, {new_path}")

        # Safely open each PDF for reading
        with open(old_path, 'rb') as f1:
            old_images = pdf_to_png(f1)
        with open(new_path, 'rb') as f2:
            new_images = pdf_to_png(f2)

        if len(old_images) != len(new_images):
            return jsonify({"error": "PDFs have different number of pages."}), 400

        result_images = []
        for page_no in range(len(old_images)):
            fixed_img = np.array(old_images[page_no])
            moving_img = np.array(new_images[page_no])
            diff_img = highlight_differences_separately(moving_img, fixed_img)
            result_images.append(Image.fromarray(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)))
        output_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        result_images[0].save(output_pdf_path, format="PDF", save_all=True, append_images=result_images[1:])
        print(f"Comparison PDF saved at {output_pdf_path.name}")
 
        # Generate download URL
        filename = os.path.basename(output_pdf_path.name)
        download_url = f"http://localhost:5000/download/{filename}"
 
        # Return the download URL as JSON
        return jsonify({"download_url": download_url}), 200
 
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return jsonify({"error": f"Error processing files: {str(e)}"}), 500
 
    finally:
        # Clean up files in static folder
        if old_path and os.path.exists(old_path):
            os.remove(old_path)
        if new_path and os.path.exists(new_path):
            os.remove(new_path)
        print("Temporary files cleaned up.")
 
# Flask endpoint to send the comparison result for download (optional)
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = safe_join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        abort(404)
    try:
        response = send_file(file_path, as_attachment=True, mimetype='application/pdf')
        # Schedule the file for deletion after sending
        @response.call_on_close
        def remove_file():
            try:
                os.unlink(file_path)
                print(f"Deleted temporary file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
        return response
    except Exception as e:
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 500
 
if __name__ == "__main__":
    app.run(debug=True)
