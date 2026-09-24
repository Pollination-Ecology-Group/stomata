import os
import argparse
import glob
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString

# Calibration
PX_TO_UM = 50.0 / 451.3958

def get_minor_axis_line(polygon):
    """
    Given a shapely Polygon, returns a LineString representing the minor axis 
    that extends well beyond the polygon.
    """
    # Get minimum area bounding rectangle
    rect = cv2.minAreaRect(np.array(polygon.exterior.coords, dtype=np.float32))
    (cx, cy), (w, h), angle = rect
    
    # We want the minor axis direction.
    # cv2.minAreaRect returns angle in [-90, 0) relative to the width.
    # if w < h, the width is the minor axis, the direction is angle
    # if w >= h, the height is the minor axis, the direction is angle - 90
    
    if w < h:
        minor_angle = angle
    else:
        minor_angle = angle - 90.0
        
    angle_rad = np.deg2rad(minor_angle)
    
    # Create a long line passing through the center along the minor axis
    length = max(w, h) * 2
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    
    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    
    return LineString([p1, p2])

def process_image(model, img_path_or_array, image_name):
    try:
        results = model(img_path_or_array, verbose=False)[0]
        
        if not results.masks:
            return []
            
        masks = results.masks.xy
        classes = results.boxes.cls.cpu().numpy()
        
        stomas = []
        centers = []
        
        for mask, cls in zip(masks, classes):
            if len(mask) < 3:
                continue
            poly = Polygon(mask)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            if int(cls) == 1:
                stomas.append(poly)
            elif int(cls) == 0:
                centers.append(poly)
                
        image_results = []
        for s_idx, stoma in enumerate(stomas):
            matched_center = None
            for center in centers:
                if stoma.contains(center.centroid):
                    matched_center = center
                    break
            
            if not matched_center and centers:
                distances = [stoma.centroid.distance(c.centroid) for c in centers]
                if min(distances) < 100:
                    matched_center = centers[np.argmin(distances)]
            
            if matched_center:
                minor_axis_line = get_minor_axis_line(stoma)
                
                stoma_intersection = minor_axis_line.intersection(stoma)
                center_intersection = minor_axis_line.intersection(matched_center)
                
                stoma_minor_length_px = stoma_intersection.length
                center_minor_length_px = center_intersection.length
                guard_cell_width_px = (stoma_minor_length_px - center_minor_length_px) / 2.0
                
                rect = cv2.minAreaRect(np.array(stoma.exterior.coords, dtype=np.float32))
                stoma_major_length_px = max(rect[1][0], rect[1][1])
                
                image_results.append({
                    "image_name": image_name,
                    "stoma_idx": s_idx,
                    "major_axis_len_px": stoma_major_length_px,
                    "major_axis_len_um": stoma_major_length_px * PX_TO_UM,
                    "minor_axis_len_px": stoma_minor_length_px,
                    "minor_axis_len_um": stoma_minor_length_px * PX_TO_UM,
                    "pore_minor_len_px": center_minor_length_px,
                    "pore_minor_len_um": center_minor_length_px * PX_TO_UM,
                    "guard_cell_width_px": guard_cell_width_px,
                    "guard_cell_width_um": guard_cell_width_px * PX_TO_UM
                })
        return image_results
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return []

def extract_traits(model_path, input_dir, output_csv):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    image_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
    
    print(f"Found {len(image_paths)} images to process.")
    
    results_list = []
    
    for img_path in image_paths:
        res = process_image(model, img_path, os.path.basename(img_path))
        results_list.extend(res)

    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract stomata geometric traits.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 segmentation model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV")
    
    args = parser.parse_args()
    extract_traits(args.model, args.input_dir, args.output_csv)
