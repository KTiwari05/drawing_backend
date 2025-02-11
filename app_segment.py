from flask import Flask, request, jsonify, make_response, send_file, abort
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
import traceback
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
from ultralytics import YOLO
import torch
from ultralytics.engine.results import Boxes
import cv2
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
 
app = Flask(__name__)
app.debug = True
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

def create_difference_masks(image1, image2):

    # Convert to grayscale
    gray_image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2GRAY)

    # Calculate differences
    diff1 = cv2.subtract(gray_image2, gray_image1)
    diff2 = cv2.subtract(gray_image1, gray_image2)

    # Threshold differences to create binary masks
    _, threshold_diff1 = cv2.threshold(diff1, 20, 255, cv2.THRESH_BINARY)
    _, threshold_diff2 = cv2.threshold(diff2, 20, 255, cv2.THRESH_BINARY)

    mask1 = cv2.merge([threshold_diff1, threshold_diff1, threshold_diff1])  # Green
    mask2 = cv2.merge([threshold_diff2, threshold_diff2, threshold_diff2])  

    green_mask = np.zeros_like(gray_image1)
    green_mask = cv2.bitwise_not(green_mask)
    green_mask = cv2.merge([green_mask,green_mask,green_mask])
    green_mask = cv2.cvtColor(green_mask, cv2.COLOR_BGR2RGB)
    green_mask[threshold_diff1 > 0] = ([0, 255, 50])
    # 
    red_mask = np.zeros_like(gray_image1)
    red_mask = cv2.bitwise_not(red_mask)
    red_mask = cv2.merge([red_mask,red_mask,red_mask])
    red_mask = cv2.cvtColor(red_mask, cv2.COLOR_BGR2RGB)
    red_mask[threshold_diff2 > 0] = ([255, 0, 50])
    # 
    

    # Invert the masks to have black over white background
    mask1 = cv2.bitwise_not(mask1)
    mask2 = cv2.bitwise_not(mask2)

    # mask1 = cv2.addWeighted(np.zeros_like(image1), 0, mask1, 1, 0) 
    # mask2 = cv2.addWeighted(np.zeros_like(image2), 0, mask1, 1, 0) 
    return mask1, mask2, green_mask, red_mask

def get_bounding_boxes(mask, model):
    bounding_boxes = []
    class_labels = []
    plotted_image = None
    table_model = YOLO("best_of_best.pt")
    other_model = YOLO("best_of_best_2.pt")

    table_results = table_model.predict(source=mask, save=False, save_txt=False)
    other_results = other_model.predict(source=mask, save=False, save_txt=False)

    results = table_results + other_results
    
    for r in table_results:
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if cls == 0:  # Assuming class 0 is the table class
                bounding_boxes.append(box)
                class_labels.append(cls)

    for r in other_results:
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if cls != 0:  # Assuming class 0 is the table class
                bounding_boxes.append(box)
                class_labels.append(cls)
    # Draw bounding boxes on the mask
    for box, cls in zip(bounding_boxes, class_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Draw rectangle
        cv2.putText(mask, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)  # Put class label

    # Display the mask with bounding boxes
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    # plt.title("Bounding Boxes with Class Labels")
    # plt.axis("off")
    # plt.show()
        # # Convert results to bounding boxes and extract visualized output
        # for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        #     bounding_boxes.append(box)  # Append bounding box coordinates
        #     class_labels.append(cls)   # Append class label

        # plotted_image = r.plot()  # Annotated image
        # r.show()

    return bounding_boxes, class_labels

def crop_bounding_boxes(image, boxes, labels):
    # print(image.shape[:2])
    cropped_regions = []
    # print(len(boxes))
    for i, box in enumerate(boxes):
        cls = labels[i]
        x1, y1, x2, y2 = map(int,box)
        cropped_regions.append((cls,image[y1:y2, x1:x2], (x1,y1,x2,y2)))
    return cropped_regions

def remove_intersections(cropped_regions, old_images):
    for i, (cls_main, main_drawing, box1) in enumerate(cropped_regions):
        if cls_main == 3:
            a, b, c, d = map(int,box1)
            image = np.array(old_images[0])
            for cls_other, crops, box2 in cropped_regions:
                if cls_other == 3:
                    continue
                x1, y1, x2, y2 = box2
                if c<x1 or a>x2 or b>y2 or d<y1:
                    continue
                roi_x1 = max(x1, a)
                roi_x2 = min(x2, c)
                roi_y1 = max(y1, b)
                roi_y2 = min(y2, d)
                
                image[roi_y1:roi_y2, roi_x1:roi_x2] = (255,255,255)

            cropped_regions[i] = ((cls_main, image[b:d, a:c], box1))
            # print(box1)

    return cropped_regions

def match_boxes_by_class(old_crops, new_crops):

    old_crops_gray = []
    new_crops_gray = []

    matches_by_class = {}
    for cls,crop_bgr,box in old_crops:
        old_crops_gray.append((cls, cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY),box))
    for cls,crop_bgr,box in new_crops:
        new_crops_gray.append((cls, cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY),box))

    for cls in set(cls for cls,_,_ in old_crops):

        cls_crop_old = [(i,cls_crops,box) for i, (cls_old, cls_crops,box) in enumerate(old_crops_gray) if cls==cls_old]
        cls_crop_new = [(j,cls_crops,box) for j, (cls_new, cls_crops,box) in enumerate(new_crops_gray) if cls==cls_new]
        

        class_matches = []

        while cls_crop_old and cls_crop_new:
            best_score = -1
            best_match = None
                
            for (i, crop_old,box_old) in cls_crop_old:
                for (j, crop_new,box_new) in cls_crop_new:

                    dimension_ratio_new = crop_new.shape[1]/crop_new.shape[0]
                    dimension_ratio_old = crop_old.shape[1]/crop_old.shape[0]
                    if abs(dimension_ratio_new - dimension_ratio_old) > 0.8:
                        continue

                    crop_new_resized = cv2.resize(crop_new, (crop_old.shape[1],crop_old.shape[0]))
                    ssim_score, _ = ssim(crop_new_resized, crop_old, full=True)

                    crop_old_flat = crop_old.flatten().reshape(1, -1)
                    crop_new_flat = crop_new_resized.flatten().reshape(1, -1)
                    cosine_score = cosine_similarity(crop_old_flat, crop_new_flat)[0, 0]
                    
                    score = 0.6*ssim_score + 0.4*cosine_score
                    # print(score)
                    if score > best_score:
                        best_score = score
                        best_match = (i, j, best_score, box_old, box_new)
            
            
            if best_match:
                index1, index2, score, box_old, box_new = best_match
                
                class_matches.append((old_crops[index1][1], new_crops[index2][1], score, box_old, box_new))

                # Remove matched boxes
                cls_crop_old = [pair for pair in cls_crop_old if pair[0] != index1]
                cls_crop_new = [pair for pair in cls_crop_new if pair[0] != index2]
            else:
                break

        matches_by_class[cls] = class_matches

    return matches_by_class

def compare_images(matches_by_class, output_green, output_red):    
    for cls, matches in reversed(matches_by_class.items()):
        for old_crop_region, new_crop_region, score, box_old, box_new in matches:
            fixed_image = np.array(old_crop_region)
            moving_image = np.array(new_crop_region)
            fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
            moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)

            fixed_image_cp = fixed_image.copy()
            moving_image_cp = moving_image.copy()

            fixed_height, fixed_width = fixed_image.shape
            moving_height, moving_width = moving_image.shape

            # Adjust the dimensions of fixed_image and moving_image to be equal using padding
            if fixed_height > moving_height:
                height_padding = fixed_height - moving_height
                pad_top = height_padding // 2
                pad_bottom = height_padding - pad_top
                moving_image = cv2.copyMakeBorder(moving_image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif moving_height > fixed_height:
                height_padding = moving_height - fixed_height
                pad_top = height_padding // 2
                pad_bottom = height_padding - pad_top
                fixed_image = cv2.copyMakeBorder(fixed_image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            if fixed_width > moving_width:
                width_padding = fixed_width - moving_width
                pad_left = width_padding // 2
                pad_right = width_padding - pad_left
                moving_image = cv2.copyMakeBorder(moving_image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif moving_width > fixed_width:
                width_padding = moving_width - fixed_width
                pad_left = width_padding // 2
                pad_right = width_padding - pad_left
                fixed_image = cv2.copyMakeBorder(fixed_image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Step 2: Detect and compute SIFT features on edge-detected images
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(fixed_image, None)
            keypoints2, descriptors2 = sift.detectAndCompute(moving_image, None)
        
            # Step 3: Match features using BFMatcher with Lowe's ratio test
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        
            # Step 4: Compute the affine transformation
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
            if len(src_pts) >= 3 :
                affine_matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts,method=cv2.RANSAC, ransacReprojThreshold=3.0)
            
            else :
                print("Not enough matches to compute affine transformation!")
                continue
            
            aligned_moving = cv2.warpAffine(moving_image, affine_matrix, (fixed_image.shape[1], fixed_image.shape[0]),flags=cv2.INTER_CUBIC)  
            aligned_moving_cp = aligned_moving.copy()


            def calculate_padding_dimensions(image):
                # Find the non-zero regions
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                rows = np.any(gray > 0, axis=1)  # Identify rows with non-black pixels
                cols = np.any(gray > 0, axis=0)  # Identify columns with non-black pixels

                # If there are no non-zero pixels, return zero padding
                if not rows.any() or not cols.any():
                    return {"top": 0, "bottom": 0, "left": 0, "right": 0}
            
                # Find the first and last non-zero rows and columns
                top_padding = np.argmax(rows)  # First row with non-black pixels
                bottom_padding = np.argmax(rows[::-1])  # Last row with non-black pixels
                left_padding = np.argmax(cols)  # First column with non-black pixels
                right_padding = np.argmax(cols[::-1])  # Last column with non-black pixels
            
        
                return {
                    "top": top_padding,
                    "bottom": bottom_padding,
                    "left": left_padding,
                    "right": right_padding
                }
            padding_dimensions_moving = calculate_padding_dimensions(aligned_moving)  
            padding_dimensions_fixed = calculate_padding_dimensions(fixed_image)  
            

            # Binarization
            if cls == 0 or cls == 1:
                _, fixed_image = cv2.threshold(fixed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, aligned_moving = cv2.threshold(aligned_moving, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
            def highlight_differences(image1, image2):
                # Convert images to grayscale if they are colored (3 channels)
                gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
                gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
            
                # # Compute the absolute difference between the two images
                # difference = cv2.absdiff(gray_image1, gray_image2)
            
                # # Optionally, apply a threshold to highlight only significant differences
                # _, threshold_diff = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
                # threshold_diff = cv2.bitwise_not(threshold_diff)
                # return threshold_diff
                # Compute differences
                diff1 = cv2.subtract(gray_image1, gray_image2)  # Things present in old but not in new
                diff2 = cv2.subtract(gray_image2, gray_image1)  # Things present in new but not in old

                # Threshold to highlight significant differences
                _, threshold_diff1 = cv2.threshold(diff1, 60, 255, cv2.THRESH_BINARY)
                _, threshold_diff2 = cv2.threshold(diff2, 60, 255, cv2.THRESH_BINARY)

                # Create a blank background image of the same size
                blank_background = np.zeros_like(gray_image2)
                
                # blank_background = cv2.cvtColor(blank_background, cv2.COLOR_BGR2RGB)
                blank_background = cv2.bitwise_not(blank_background)
                green_mask = cv2.merge([blank_background, blank_background, blank_background])
                red_mask = cv2.merge([blank_background, blank_background, blank_background])
                # Highlight differences on the blank background
                # blank_background[threshold_diff1 > 0] = [0, 255, 0]
                green_mask[threshold_diff2 > 0] = [0, 255, 0]
                green_mask = cv2.cvtColor(green_mask, cv2.COLOR_BGR2RGB)
                red_mask[threshold_diff2 > 0] = [100, 0, 200]
                red_mask = cv2.cvtColor(red_mask, cv2.COLOR_BGR2RGB)
            
                return green_mask, red_mask
            
            # Highlight differences between cropped_fixed_image and cropped_moving_image
            
            green_mask, red_mask = highlight_differences(fixed_image, aligned_moving)

            
            def intersection_image(green_mask, red_mask, padding_dimensions_fixed, padding_dimensions_moving):

                base_green = np.zeros_like(green_mask)
                base_red = np.zeros_like(red_mask)
                # Extract padding values
                top_pad = max(padding_dimensions_fixed["top"], padding_dimensions_moving["top"])
                bottom_pad = max(padding_dimensions_fixed["bottom"], padding_dimensions_moving["bottom"])
                left_pad = max(padding_dimensions_fixed["left"], padding_dimensions_moving["left"])
                right_pad = max(padding_dimensions_fixed["right"], padding_dimensions_moving["right"])
                top = top_pad
                bottom = base_green.shape[0] - bottom_pad
                left = left_pad
                right = base_green.shape[1] - right_pad
                
                base_green[top:bottom, left:right] = green_mask[top:bottom, left:right]
                base_red[top:bottom, left:right] = red_mask[top:bottom, left:right]
                # Perform cropping
                return base_green, base_red
            
            
            intersection_green, intersection_red = intersection_image(green_mask, red_mask, padding_dimensions_fixed, padding_dimensions_moving)


            def crop_image(image, padding_dimensions):
        
                # Extract padding values
                top = padding_dimensions["top"]
                bottom = image.shape[0] - padding_dimensions["bottom"]
                left = padding_dimensions["left"]
                right = image.shape[1] - padding_dimensions["right"]
                
                # Perform cropping
                return image[top:bottom, left:right]
            
            def cropping(intersection_green, intersection_red, padding_dimensions_fixed, padding_dimensions_moving):
                padded_green = crop_image(intersection_green, padding_dimensions_fixed)
                padded_red = crop_image(intersection_red, padding_dimensions_moving)
                return padded_green, padded_red
            
            padded_green, padded_red = cropping(intersection_green, intersection_red, padding_dimensions_fixed, padding_dimensions_moving)
            aligned_moving_cp = crop_image(aligned_moving_cp, padding_dimensions_moving)
            
                                            
            
            def replace(background, image, box):
                padding_dimensions = calculate_padding_dimensions(image)
                
                # Extract padding values
                top = padding_dimensions["top"]
                bottom = image.shape[0] - padding_dimensions["bottom"]
                left = padding_dimensions["left"]
                right = image.shape[1] - padding_dimensions["right"]
                
                (x1, y1, x2, y2) = box
                non_padded = image[top:bottom, left:right]
                background[y1 + top : y1 + top + non_padded.shape[0], x1 + left : x1 + left + non_padded.shape[1]] = non_padded
                
                return background
                # temp = background[y1:y2, x1:x2][:]
                # print(x1, y1, x2, y2)
            output_green = replace(output_green, padded_green, box_old)
            output_red = replace(output_red, padded_red, box_new)

    return output_green, output_red    
        
def align_images(image1, image2):
    """Aligns two images using ORB feature matching and homography."""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    aspect_ratio1 = w1 / h1
    aspect_ratio2 = w2 / h2

    if (aspect_ratio1 == aspect_ratio2) :
        return image1, image2
    
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

def save_images_as_pdf(images, output_pdf_path):
    rgb_images = [img.convert('RGB') for img in images]
    rgb_images[0].save(output_pdf_path, save_all=True, append_images=rgb_images[1:], resolution=100.0)


# PDF comparison route
@app.route('/compare', methods=['POST'])
@jwt_required()
def compare_files():
    model1 = YOLO("best_of_best_2.pt")
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
            old_image = np.array(old_images[page_no])
            new_image = np.array(new_images[page_no])
            aligned_image1, aligned_image2 = align_images(old_image, new_image)
            # aligned_image1, aligned_image2 = old_image, new_image
            mask1, mask2, green_mask_bg, red_mask_bg = create_difference_masks(aligned_image1, aligned_image2)
            old_boxes, old_classes = get_bounding_boxes(mask1, model1)
            new_boxes, new_classes = get_bounding_boxes(mask2, model1)
            old_crops = crop_bounding_boxes(np.array(aligned_image1), old_boxes, old_classes)
            new_crops = crop_bounding_boxes(np.array(aligned_image2), new_boxes, new_classes)
            matches_by_class = match_boxes_by_class(old_crops, new_crops)
            
            pil_images = []
            output_green = green_mask_bg.copy()
            output_red = red_mask_bg.copy()
            output_green, output_red = compare_images(matches_by_class, output_green, output_red)
            
            original_gray = cv2.cvtColor(np.array(aligned_image2), cv2.COLOR_RGB2GRAY)
            original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)

            mask_green = np.all(output_green == np.array([0,255,50]), axis=-1)
            original_rgb[mask_green] = [0,255,50]
            mask_red = np.all(output_red == np.array([255,0,50]), axis=-1)
            original_rgb[mask_red] = [255,0,50]
            # original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(original_rgb)
            result_images.append(result_image)

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
    app.run(host='0.0.0.0',debug=True)
