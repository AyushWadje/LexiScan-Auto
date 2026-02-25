"""
Spatial OCR Demo - Shows Advanced Features
============================================
This demonstrates:
1. Extracting word locations and confidence scores
2. Filtering by confidence threshold
3. Reconstructing text while preserving layout
4. Drawing bounding boxes for visualization
"""

import sys
import os

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from pathlib import Path
import pandas as pd
from lexiscan.ocr.pipeline import (
    process_pdf_spatial,
    filter_by_confidence,
    reconstruct_text_from_spatial,
    visualize_spatial_ocr,
    convert_page,
    preprocess_image,
    ocr_image_spatial
)

def demo_spatial_extraction():
    """Demo 1: Extract and analyze spatial data"""
    print("\n" + "="*70)
    print("DEMO 1: Spatial Data Extraction")
    print("="*70 + "\n")
    
    pdf_path = r"C:\Users\wadje\Downloads\AyushWadjeResume.pdf"
    
    # Extract spatial data (all words)
    spatial_data = process_pdf_spatial(pdf_path, confidence_threshold=0)
    
    for page_num, df in spatial_data.items():
        print(f"\n📄 Page {page_num} Statistics:")
        print(f"   Total words: {len(df)}")
        print(f"   Average confidence: {df['conf'].mean():.1f}%")
        print(f"   Confidence range: {df['conf'].min():.0f}% - {df['conf'].max():.0f}%")
        
        # Show top 10 words by position (left-most, top-most)
        print(f"\n   First 10 words on page:")
        top_words = df.nsmallest(10, 'top')
        for idx, row in top_words.iterrows():
            print(f"     '{row['text']}' @ ({row['left']}, {row['top']}) - conf: {row['conf']:.0f}%")


def demo_confidence_filtering():
    """Demo 2: Filter words by confidence threshold"""
    print("\n" + "="*70)
    print("DEMO 2: Confidence Filtering")
    print("="*70 + "\n")
    
    pdf_path = r"C:\Users\wadje\Downloads\AyushWadjeResume.pdf"
    
    # Extract all words
    spatial_data = process_pdf_spatial(pdf_path, confidence_threshold=0)
    
    thresholds = [0, 50, 70, 90]
    
    for page_num, df_all in spatial_data.items():
        print(f"\n📊 Page {page_num} - Words by Confidence Threshold:")
        
        for threshold in thresholds:
            df_filtered = filter_by_confidence(df_all, threshold=threshold)
            percentage = (len(df_filtered) / len(df_all)) * 100
            print(f"   ≥ {threshold}%: {len(df_filtered):3d} words ({percentage:.1f}%)")


def demo_layout_reconstruction():
    """Demo 3: Reconstruct text preserving layout"""
    print("\n" + "="*70)
    print("DEMO 3: Layout-Preserving Text Reconstruction")
    print("="*70 + "\n")
    
    pdf_path = r"C:\Users\wadje\Downloads\AyushWadjeResume.pdf"
    
    spatial_data = process_pdf_spatial(pdf_path, confidence_threshold=60)
    
    for page_num, df in spatial_data.items():
        print(f"\n📄 Page {page_num} - Reconstructed Text (60%+ confidence):\n")
        reconstructed = reconstruct_text_from_spatial(df)
        # Show first 500 characters
        preview = reconstructed[:500]
        print(preview)
        if len(reconstructed) > 500:
            print("\n[...truncated...]")


def demo_advanced_filtering():
    """Demo 4: Advanced filtering - e.g., exclude low-confidence words"""
    print("\n" + "="*70)
    print("DEMO 4: Advanced Filtering Strategies")
    print("="*70 + "\n")
    
    pdf_path = r"C:\Users\wadje\Downloads\AyushWadjeResume.pdf"
    
    spatial_data = process_pdf_spatial(pdf_path, confidence_threshold=0)
    
    for page_num, df in spatial_data.items():
        print(f"\n📄 Page {page_num} - Custom Filtering:\n")
        
        # Strategy 1: Only alphabetic words (no noise)
        alpha_only = df[df['text'].str.replace('[^a-zA-Z]', '', regex=True).str.len() > 0]
        print(f"Alphabetic words only: {len(alpha_only)} (removed {len(df) - len(alpha_only)} symbols/numbers)")
        
        # Strategy 2: Exclude single-character words (often noise)
        multi_char = df[df['text'].str.len() > 1]
        print(f"Multi-character words: {len(multi_char)} (removed {len(df) - len(multi_char)} single chars)")
        
        # Strategy 3: Focus on main text area (exclude margins)
        main_area = df[(df['left'] > 100) & (df['left'] < 1500)]
        print(f"Main text area (x=100-1500): {len(main_area)} words")
        
        # Strategy 4: High confidence AND alphabetic
        quality = df[(df['conf'] >= 70) & (df['text'].str.replace('[^a-zA-Z]', '', regex=True).str.len() > 0)]
        print(f"High quality (70%+ conf + alphabetic): {len(quality)} words")


def demo_csv_export():
    """Demo 5: Export spatial data to CSV for analysis"""
    print("\n" + "="*70)
    print("DEMO 5: CSV Export for Further Analysis")
    print("="*70 + "\n")
    
    pdf_path = r"C:\Users\wadje\Downloads\AyushWadjeResume.pdf"
    
    spatial_data = process_pdf_spatial(pdf_path, confidence_threshold=30)
    
    # Combine all pages
    all_data = []
    for page_num, df in spatial_data.items():
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Export to CSV
    csv_file = "spatial_ocr_demo.csv"
    combined_df.to_csv(csv_file, index=False)
    
    print(f"\n✓ Exported {len(combined_df)} words to: {csv_file}")
    print(f"\nColumns available for analysis:")
    for i, col in enumerate(combined_df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nDataFrame shape: {combined_df.shape}")
    print(f"\nConfidence statistics:")
    print(f"  Mean: {combined_df['conf'].mean():.1f}%")
    print(f"  Median: {combined_df['conf'].median():.1f}%")
    print(f"  Std Dev: {combined_df['conf'].std():.1f}%")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█  SPATIAL OCR DEMONSTRATION")
    print("█"*70)
    
    # Run all demos
    demo_spatial_extraction()
    demo_confidence_filtering()
    demo_layout_reconstruction()
    demo_advanced_filtering()
    demo_csv_export()
    
    print("\n" + "█"*70)
    print("█  DEMOS COMPLETE")
    print("█"*70 + "\n")
