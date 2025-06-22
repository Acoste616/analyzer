"""Enhanced image extractor with real OCR and AI analysis."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import exifread
import torch

from .base_extractor import BaseExtractor, ExtractedContent
from ..core.logger import get_logger
from ..processors.lm_studio import LMStudioProcessor

logger = get_logger(__name__)


class EnhancedImageExtractor(BaseExtractor):
    """Enhanced image extractor with multiple analysis methods."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    def __init__(self):
        super().__init__()
        self.lm_studio = LMStudioProcessor()
        self._init_ocr_engines()
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _init_ocr_engines(self):
        """Initialize OCR engines (EasyOCR and Tesseract)."""
        self._easyocr_available = False
        self._tesseract_available = False
        self._easyocr_reader = None
        
        # Initialize EasyOCR
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en', 'pl'], gpu=torch.cuda.is_available())
            self._easyocr_available = True
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")
        
        # Initialize Tesseract with PATH configuration
        try:
            import pytesseract
            from PIL import Image
            
            # Configure Tesseract executable path for Windows
            import os
            import platform
            if platform.system() == "Windows":
                # Common Tesseract installation paths
                tesseract_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Tesseract-OCR\tesseract.exe"
                ]
                
                for path in tesseract_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Tesseract configured at: {path}")
                        break
                else:
                    # Try to find in PATH
                    import shutil
                    tesseract_path = shutil.which("tesseract")
                    if tesseract_path:
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                        logger.info(f"Tesseract found in PATH: {tesseract_path}")
            
            # Test Tesseract
            test_img = Image.new('RGB', (50, 20), color='white')
            pytesseract.image_to_string(test_img)
            self._tesseract_available = True
            logger.info("Tesseract OCR initialized successfully")
            
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            logger.info("Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    
    async def extract(self, file_path: str) -> ExtractedContent:
        """Extract text and analyze content from image with comprehensive testing."""
        logger.info(f"Starting enhanced image extraction for: {file_path}")
        
        try:
            # Load and validate image
            image = Image.open(file_path)
            image = image.convert('RGB')
            
            # Test OCR engines before processing
            if not hasattr(self, '_ocr_tested'):
                logger.info("🧪 Testing OCR engines...")
                test_results = self._test_ocr_engines()
                self._ocr_tested = True
                
                if not any(test_results.values()):
                    logger.error("❌ Both OCR engines failed tests!")
                    return ExtractedContent(
                        title=f"OCR Error - {Path(file_path).stem}",
                        summary="Błąd: OCR engines nie działają",
                        content="BŁĄD: OCR engines nie działają prawidłowo",
                        metadata={"error": "OCR engines failed testing"},
                        confidence=0.0,
                        language="pl"
                    )
            
            # Analyze image content first
            content_analysis = await self._analyze_image_content(image)
            
            # Extract text using OCR
            extracted_text = await self._extract_text_ocr(file_path, image)
            
            # Determine if extraction was successful
            has_meaningful_text = (
                extracted_text and 
                len(extracted_text.strip()) > 5 and
                not extracted_text.startswith("OCR nie wykrył") and
                not extracted_text.startswith("Brak treści")
            )
            
            if has_meaningful_text:
                # Successful text extraction
                logger.info(f"✅ Extracted {len(extracted_text)} characters from image")
                confidence = min(0.7 + content_analysis['complexity_score'] * 0.3, 0.95)
                
                content = f"[{content_analysis['content_type'].upper()}]\n{extracted_text}"
                
            elif content_analysis['has_text'] and content_analysis['text_regions'] > 5:
                # Image has text but OCR failed
                content = f"Obraz zawiera tekst ({content_analysis['text_regions']} regionów), ale OCR nie zdołał go odczytać. Możliwe przyczyny: zła jakość, mały rozmiar tekstu, lub nieczytelny font."
                confidence = 0.3
                
            elif content_analysis['content_type'] in ['chart_diagram', 'diagram']:
                # Visual content without text
                content = f"Materiał wizualny: {content_analysis['content_type']} z {content_analysis.get('text_regions', 0)} elementami tekstowymi. Zawiera diagramy, wykresy lub schematy edukacyjne."
                confidence = 0.6
                
            else:
                # Likely a photo or image without educational text
                content = f"Zdjęcie lub obraz bez tekstu edukacyjnego. Typ: {content_analysis['content_type']}. Możliwe zastosowanie jako materiał ilustracyjny."
                confidence = 0.4
            
            # Generate metadata
            metadata = await self._extract_metadata(file_path, image)
            metadata.update({
                'content_analysis': content_analysis,
                'extraction_method': 'enhanced_ocr_analysis',
                'ocr_engines_used': [
                    engine for engine, available in [
                        ('tesseract', self._tesseract_available),
                        ('easyocr', self._easyocr_available)
                    ] if available
                ]
            })
            
            # Generate summary based on content
            if has_meaningful_text:
                summary = f"Materiał edukacyjny z {len(extracted_text)} znakami tekstu."
            elif content_analysis['content_type'] in ['chart_diagram', 'diagram']:
                summary = f"Diagram edukacyjny z {content_analysis.get('text_regions', 0)} elementami."
            else:
                summary = f"Materiał wizualny typu {content_analysis['content_type']}."
            
            return ExtractedContent(
                title=Path(file_path).stem,
                summary=summary,
                content=content,
                metadata=metadata,
                confidence=confidence,
                language="pl"
            )
            
        except Exception as e:
            logger.error(f"Enhanced image extraction failed for {file_path}: {e}")
            return ExtractedContent(
                title=f"Error - {Path(file_path).stem}",
                summary=f"Błąd przetwarzania: {str(e)}",
                content=f"Błąd przetwarzania obrazu: {str(e)}",
                metadata={"error": str(e), "file_path": file_path},
                confidence=0.0,
                language="pl"
            )
    
    async def extract_with_memory_management(self, file_path: str) -> ExtractedContent:
        """Extract with automatic memory management"""
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = await self.extract(file_path)
            
            # Clear cache after processing large files
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > 50 and torch.cuda.is_available():  # Lower threshold for images
                torch.cuda.empty_cache()
                
            return result
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"GPU OOM for {file_path}, falling back to CPU")
            torch.cuda.empty_cache()
            
            # Retry with CPU (EasyOCR fallback)
            result = await self.extract(file_path)
            return result
    
    async def _extract_metadata(self, file_path: str, image: Image) -> Dict[str, Any]:
        """Extract image metadata."""
        metadata = self.get_file_info(file_path)
        metadata.update({
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "has_transparency": image.mode in ('RGBA', 'LA'),
        })
        
        # Extract EXIF data
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                if tags:
                    metadata["camera"] = str(tags.get('Image Make', '')).strip()
                    metadata["model"] = str(tags.get('Image Model', '')).strip()
                    metadata["datetime"] = str(tags.get('EXIF DateTimeOriginal', '')).strip()
                    metadata["gps"] = bool(tags.get('GPS GPSLatitude'))
        except Exception as e:
            logger.debug(f"EXIF extraction failed: {e}")
        
        return metadata
    
    async def _extract_text_ocr(self, file_path: str, image: Image) -> str:
        """Extract text using multiple OCR engines with enhanced processing."""
        extracted_texts = []
        
        # Try EasyOCR first (usually better for varied text)
        if self._easyocr_available and self._easyocr_reader:
            try:
                logger.debug(f"Running EasyOCR on {file_path}")
                # Convert PIL Image to numpy array for EasyOCR
                img_array = np.array(image)
                
                # Run EasyOCR with confidence threshold
                results = self._easyocr_reader.readtext(img_array, detail=0, paragraph=True)
                if results:
                    easyocr_text = " ".join(results).strip()
                    if easyocr_text and len(easyocr_text) > 3:
                        extracted_texts.append(("EasyOCR", easyocr_text))
                        logger.info(f"EasyOCR extracted {len(easyocr_text)} characters")
                
            except Exception as e:
                logger.warning(f"EasyOCR processing failed: {e}")
        
        # Try Tesseract with proper configuration
        if self._tesseract_available:
            try:
                import pytesseract
                import os
                
                # Ensure Tesseract path is set
                if not pytesseract.pytesseract.tesseract_cmd or pytesseract.pytesseract.tesseract_cmd == 'tesseract':
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                
                logger.debug(f"Running Tesseract on {file_path}")
                
                # Preprocess image for better OCR
                processed_image = self._preprocess_image_for_ocr(image)
                
                # Multiple Tesseract configurations for different text types
                tesseract_configs = [
                    '--oem 3 --psm 6 -l pol+eng',  # Default mixed text
                    '--oem 3 --psm 8 -l pol+eng',  # Single word
                    '--oem 3 --psm 7 -l pol+eng',  # Single text line
                    '--oem 3 --psm 11 -l pol+eng', # Sparse text
                    '--oem 3 --psm 12 -l pol+eng', # Sparse text with OSD
                ]
                
                best_result = ""
                best_confidence = 0
                
                for config in tesseract_configs:
                    try:
                        # Try with original image
                        text = pytesseract.image_to_string(image, config=config).strip()
                        if len(text) > len(best_result):
                            best_result = text
                        
                        # Try with processed image
                        text_processed = pytesseract.image_to_string(processed_image, config=config).strip()
                        if len(text_processed) > len(best_result):
                            best_result = text_processed
                            
                    except Exception as e:
                        logger.debug(f"Tesseract config {config} failed: {e}")
                        continue
                
                if best_result and len(best_result) > 3:
                    extracted_texts.append(("Tesseract", best_result))
                    logger.info(f"Tesseract extracted {len(best_result)} characters")
                
            except Exception as e:
                logger.warning(f"Tesseract processing failed: {e}")
        
        # Combine and clean results
        if extracted_texts:
            # Choose the longest/best result
            best_text = max(extracted_texts, key=lambda x: len(x[1]))
            final_text = self._clean_ocr_text(best_text[1])
            
            logger.info(f"Final OCR result from {best_text[0]}: {len(final_text)} characters")
            return final_text
        
        logger.warning(f"No text extracted from {file_path}")
        return ""
    
    def _preprocess_image_for_ocr(self, image: Image) -> Image:
        """Preprocess image to improve OCR accuracy."""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to CV2
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply image processing techniques
            # 1. Resize if too small (OCR works better on larger images)
            height, width = gray.shape
            if height < 100 or width < 100:
                scale_factor = max(200 / height, 200 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 2. Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 3. Thresholding to get binary image
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            return Image.fromarray(cleaned)
            
        except Exception as e:
            logger.debug(f"Image preprocessing failed: {e}")
            return image
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR text."""
        if not text:
            return ""
        
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\\]', '', text)
        
        # Fix common OCR mistakes (add more as needed)
        replacements = {
            'rn': 'm',
            'cl': 'd',
            '0': 'o',  # only in words, not numbers
            '1': 'l',  # only in words
        }
        
        # Apply replacements carefully (only in word contexts)
        words = text.split()
        cleaned_words = []
        for word in words:
            # Skip if word looks like a number
            if word.isdigit():
                cleaned_words.append(word)
                continue
                
            cleaned_word = word
            for wrong, correct in replacements.items():
                if wrong in cleaned_word and not cleaned_word.isdigit():
                    cleaned_word = cleaned_word.replace(wrong, correct)
            cleaned_words.append(cleaned_word)
        
        return ' '.join(cleaned_words).strip()
    
    async def _analyze_image_content(self, image: Image) -> Dict[str, Any]:
        """Analyze image to determine if it contains text, diagrams, or other content."""
        content_analysis = {
            'has_text': False,
            'has_diagrams': False,
            'has_charts': False,
            'content_type': 'unknown',
            'text_regions': 0,
            'complexity_score': 0.0
        }
        
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to CV2
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect text regions using CV2
            # Edge detection for text areas
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Text-sized areas
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 10:  # Text-like aspect ratios
                        text_like_contours += 1
            
            content_analysis['text_regions'] = text_like_contours
            content_analysis['has_text'] = text_like_contours > 5
            
            # Detect line patterns (diagrams, charts)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            if lines is not None:
                content_analysis['has_diagrams'] = len(lines) > 10
                content_analysis['has_charts'] = len(lines) > 20
            
            # Calculate complexity score
            complexity = (text_like_contours * 0.3 + 
                         (len(lines) if lines is not None else 0) * 0.1 + 
                         len(contours) * 0.01)
            content_analysis['complexity_score'] = min(complexity / 100, 1.0)
            
            # Determine content type
            if content_analysis['has_text'] and text_like_contours > 20:
                content_analysis['content_type'] = 'text_document'
            elif content_analysis['has_charts']:
                content_analysis['content_type'] = 'chart_diagram'
            elif content_analysis['has_diagrams']:
                content_analysis['content_type'] = 'diagram'
            elif content_analysis['has_text']:
                content_analysis['content_type'] = 'mixed_content'
            else:
                content_analysis['content_type'] = 'photo_image'
            
            logger.debug(f"Image analysis: {content_analysis}")
            
        except Exception as e:
            logger.warning(f"Image content analysis failed: {e}")
        
        return content_analysis
    
    def _test_ocr_engines(self) -> Dict[str, bool]:
        """Test OCR engines with real text to ensure they work."""
        test_results = {
            'tesseract': False,
            'easyocr': False
        }
        
        try:
            # Create a test image with clear text
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            # Create test image with Polish and English text
            test_img = Image.new('RGB', (400, 200), color='white')
            draw = ImageDraw.Draw(test_img)
            
            # Try to use a better font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Draw test text
            test_text = "Test tekst 123\nPOLISH TEXT\nEducation"
            draw.text((20, 50), test_text, fill='black', font=font)
            
            # Test Tesseract
            if self._tesseract_available:
                try:
                    import pytesseract
                    result = pytesseract.image_to_string(test_img, config='--oem 3 --psm 6 -l pol+eng')
                    if 'test' in result.lower() or 'tekst' in result.lower():
                        test_results['tesseract'] = True
                        logger.info("✅ Tesseract OCR test passed")
                    else:
                        logger.warning(f"⚠️ Tesseract OCR test failed: {repr(result)}")
                except Exception as e:
                    logger.error(f"❌ Tesseract test error: {e}")
            
            # Test EasyOCR
            if self._easyocr_available and self._easyocr_reader:
                try:
                    img_array = np.array(test_img)
                    results = self._easyocr_reader.readtext(img_array, detail=0)
                    combined_text = " ".join(results).lower()
                    if 'test' in combined_text or 'tekst' in combined_text:
                        test_results['easyocr'] = True
                        logger.info("✅ EasyOCR test passed")
                    else:
                        logger.warning(f"⚠️ EasyOCR test failed: {repr(results)}")
                except Exception as e:
                    logger.error(f"❌ EasyOCR test error: {e}")
            
        except Exception as e:
            logger.error(f"OCR testing failed: {e}")
        
        return test_results 