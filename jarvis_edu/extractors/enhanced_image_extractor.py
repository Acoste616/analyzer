"""Enhanced image extractor with real OCR and AI analysis."""

import asyncio
import hashlib
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
        self._ocr_cache = {}
        self._cache_dir = Path("cache/ocr_results")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_ocr_engines()
    
    def supports_file(self, file_path: str) -> bool:
        """Check if this extractor supports the file type."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _init_ocr_engines(self):
        """Initialize OCR engines (EasyOCR, Tesseract, and Cloud OCR)."""
        self._easyocr_available = False
        self._tesseract_available = False
        self._cloud_ocr_available = False
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
        
        # Initialize Cloud OCR (optional, for difficult cases)
        try:
            # Check if cloud OCR dependencies are available
            # This is a placeholder - you can add actual cloud OCR services like Google Vision API
            self._cloud_ocr_available = False  # Set to True when cloud OCR is configured
            if self._cloud_ocr_available:
                logger.info("Cloud OCR initialized successfully")
        except Exception as e:
            logger.debug(f"Cloud OCR not configured: {e}")
    
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
            
            # Check cache first, then extract text using OCR
            cached_text = await self._get_cached_ocr(file_path)
            if cached_text is not None:
                logger.info(f"Using cached OCR result for {file_path}")
                extracted_text = cached_text
            else:
                extracted_text = await self._extract_text_ocr(file_path, image)
                # Cache the result if it's meaningful
                if extracted_text and len(extracted_text.strip()) > 5:
                    await self._cache_ocr_result(file_path, extracted_text)
            
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
        """Multi-engine OCR with fallback and validation"""
        results = []
        
        # Preprocess image for better OCR
        preprocessed_variants = await self._preprocess_for_ocr(image)
        
        # 1. Try EasyOCR first (best for mixed content)
        if self._easyocr_available and self._easyocr_reader:
            for i, variant in enumerate(preprocessed_variants):
                try:
                    easyocr_text = await self._run_easyocr(variant)
                    if self._validate_ocr_result(easyocr_text):
                        confidence = self._calculate_confidence(easyocr_text)
                        results.append(('easyocr', easyocr_text, confidence))
                        logger.debug(f"EasyOCR variant {i}: {len(easyocr_text)} chars, confidence {confidence}")
                except Exception as e:
                    logger.debug(f"EasyOCR variant {i} failed: {e}")
        
        # 2. Try Tesseract with multiple configs
        if self._tesseract_available:
            configs = self._get_tesseract_configs()
            for variant in preprocessed_variants:
                for config in configs:
                    try:
                        tess_text = await self._run_tesseract(variant, config)
                        if self._validate_ocr_result(tess_text):
                            confidence = self._calculate_confidence(tess_text)
                            results.append(('tesseract', tess_text, confidence))
                            logger.debug(f"Tesseract {config}: {len(tess_text)} chars, confidence {confidence}")
                    except Exception as e:
                        logger.debug(f"Tesseract {config} failed: {e}")
        
        # 3. Try cloud OCR as last resort for difficult images
        if not results and self._cloud_ocr_available:
            try:
                cloud_text = await self._run_cloud_ocr(image)
                if cloud_text:
                    results.append(('cloud', cloud_text, 0.7))
                    logger.debug(f"Cloud OCR: {len(cloud_text)} chars")
            except Exception as e:
                logger.warning(f"Cloud OCR failed: {e}")
        
        # Return best result
        if results:
            # Sort by confidence
            results.sort(key=lambda x: x[2], reverse=True)
            best_engine, best_text, confidence = results[0]
            final_text = self._clean_ocr_text(best_text)
            logger.info(f"Best OCR result from {best_engine} with confidence {confidence:.2f}: {len(final_text)} chars")
            return final_text
        
        logger.warning(f"No valid OCR results for {file_path}")
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
    
    # === Cache Management Methods ===
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image"""
        stat = Path(image_path).stat()
        return hashlib.md5(f"{image_path}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()

    async def _get_cached_ocr(self, image_path: str) -> Optional[str]:
        """Get cached OCR result if available"""
        cache_key = self._get_cache_key(image_path)
        cache_file = self._cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.debug(f"Failed to read cache file: {e}")
        return None

    async def _cache_ocr_result(self, image_path: str, text: str):
        """Cache OCR result for future use"""
        try:
            cache_key = self._get_cache_key(image_path)
            cache_file = self._cache_dir / f"{cache_key}.txt"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.debug(f"Cached OCR result for {Path(image_path).name}")
        except Exception as e:
            logger.debug(f"Failed to cache OCR result: {e}")
    
    # === OCR Engine Methods ===
    
    async def _run_easyocr(self, image: Image) -> str:
        """Run EasyOCR on image"""
        if not self._easyocr_available or not self._easyocr_reader:
            return ""
        
        # Convert PIL Image to numpy array for EasyOCR
        img_array = np.array(image)
        
        # Run EasyOCR with confidence threshold
        results = self._easyocr_reader.readtext(img_array, detail=0, paragraph=True)
        if results:
            return " ".join(results).strip()
        return ""
    
    async def _run_tesseract(self, image: Image, config: str) -> str:
        """Run Tesseract OCR with specific config"""
        if not self._tesseract_available:
            return ""
        
        try:
            import pytesseract
            
            # Ensure Tesseract path is set
            if not pytesseract.pytesseract.tesseract_cmd or pytesseract.pytesseract.tesseract_cmd == 'tesseract':
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            text = pytesseract.image_to_string(image, config=config).strip()
            return text
        except Exception as e:
            logger.debug(f"Tesseract with config {config} failed: {e}")
            return ""
    
    async def _run_cloud_ocr(self, image: Image) -> str:
        """Run cloud OCR service (placeholder for future implementation)"""
        # This is a placeholder for cloud OCR services like Google Vision API
        # You can implement specific cloud OCR services here
        logger.debug("Cloud OCR not implemented yet")
        return ""
    
    def _get_tesseract_configs(self) -> List[str]:
        """Get list of Tesseract configurations to try"""
        return [
            '--oem 3 --psm 6 -l pol+eng',  # Default mixed text
            '--oem 3 --psm 8 -l pol+eng',  # Single word
            '--oem 3 --psm 7 -l pol+eng',  # Single text line
            '--oem 3 --psm 11 -l pol+eng', # Sparse text
            '--oem 3 --psm 12 -l pol+eng', # Sparse text with OSD
        ]
    
    def _validate_ocr_result(self, text: str) -> bool:
        """Validate OCR result quality"""
        if not text or len(text) < 5:
            return False
        
        # Check for garbage characters
        valid_chars = len([c for c in text if c.isalnum() or c.isspace() or c in '.,!?-:;()[]{}'])
        if valid_chars / len(text) < 0.7:
            return False
        
        # Check for repeating patterns (OCR artifacts)
        words = text.split()
        if len(words) > 3 and len(set(words)) < len(words) * 0.3:
            return False
        
        return True
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for OCR result"""
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length bonus (longer text usually means better OCR)
        if len(text) > 50:
            score += 0.2
        elif len(text) > 20:
            score += 0.1
        
        # Word diversity bonus
        words = text.split()
        if len(words) > 0:
            word_diversity = len(set(words)) / len(words)
            score += word_diversity * 0.2
        
        # Character quality bonus
        valid_chars = len([c for c in text if c.isalnum() or c.isspace() or c in '.,!?-:;()[]{}'])
        char_quality = valid_chars / len(text)
        score += char_quality * 0.1
        
        return min(score, 1.0)
    
    # === Image Preprocessing Methods ===
    
    async def _preprocess_for_ocr(self, image: Image) -> List[Image]:
        """Generate multiple preprocessed versions for better OCR"""
        variants = [image]  # Original
        
        try:
            # Detect image type
            img_type = self._detect_image_type(image)
            
            if img_type == 'screenshot':
                # For screenshots - enhance contrast
                enhanced = self._enhance_screenshot(image)
                if enhanced:
                    variants.append(enhanced)
            
            elif img_type == 'photo_of_text':
                # For photos - deskew, denoise
                processed = self._process_photo_text(image)
                if processed:
                    variants.append(processed)
            
            elif img_type == 'handwritten':
                # For handwritten - special processing
                processed = self._process_handwritten(image)
                if processed:
                    variants.append(processed)
            
            elif img_type == 'diagram':
                # For diagrams - isolate text regions
                text_regions = self._extract_text_regions(image)
                variants.extend(text_regions)
            
            # Always add a basic preprocessed version
            basic_processed = self._preprocess_image_for_ocr(image)
            if basic_processed and basic_processed != image:
                variants.append(basic_processed)
                
        except Exception as e:
            logger.debug(f"Image preprocessing failed: {e}")
        
        return variants
    
    def _detect_image_type(self, image: Image) -> str:
        """Detect the type of image for appropriate preprocessing"""
        try:
            import cv2
            import numpy as np
            
            # Convert to array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Classify based on characteristics
            if mean_intensity > 200 and std_intensity < 50:
                return 'screenshot'  # High brightness, low variation
            elif edge_density > 0.1:
                return 'diagram'  # Many edges
            elif std_intensity > 80:
                return 'photo_of_text'  # High variation (photo)
            elif mean_intensity < 150:
                return 'handwritten'  # Darker, potentially handwritten
            else:
                return 'mixed_content'
                
        except Exception as e:
            logger.debug(f"Image type detection failed: {e}")
            return 'mixed_content'
    
    def _enhance_screenshot(self, image: Image) -> Optional[Image]:
        """Enhance screenshot for better OCR"""
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return Image.fromarray(sharpened)
            
        except Exception as e:
            logger.debug(f"Screenshot enhancement failed: {e}")
            return None
    
    def _process_photo_text(self, image: Image) -> Optional[Image]:
        """Process photo of text (deskew, denoise)"""
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return Image.fromarray(adaptive_thresh)
            
        except Exception as e:
            logger.debug(f"Photo text processing failed: {e}")
            return None
    
    def _process_handwritten(self, image: Image) -> Optional[Image]:
        """Process handwritten text"""
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Morphological operations to connect broken characters
            kernel = np.ones((2,2), np.uint8)
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
            )
            
            return Image.fromarray(thresh)
            
        except Exception as e:
            logger.debug(f"Handwritten processing failed: {e}")
            return None
    
    def _extract_text_regions(self, image: Image) -> List[Image]:
        """Extract text regions from diagrams"""
        regions = []
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Find text-like contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Text-sized areas
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 10:  # Text-like aspect ratios
                        # Extract region with padding
                        pad = 5
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(gray.shape[1], x + w + pad)
                        y2 = min(gray.shape[0], y + h + pad)
                        
                        region = gray[y1:y2, x1:x2]
                        if region.size > 0:
                            regions.append(Image.fromarray(region))
            
        except Exception as e:
            logger.debug(f"Text region extraction failed: {e}")
        
        return regions[:5]  # Limit to 5 regions to avoid processing too many
    
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