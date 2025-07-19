import os
import re
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# === CONFIG ===
PDF_PATH = "./company_docs/KGT Solutions - Company Data.pdf"
INDEX_PATH = "./company_docs/index.faiss"
CHUNKS_PATH = "./company_docs/chunks.pkl"
METADATA_PATH = "./company_docs/metadata.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100  # Increased overlap to preserve context
EMBED_MODEL = 'all-MiniLM-L6-v2'

class EnhancedPDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
    
    def extract_text_with_structure(self, path: str) -> List[Dict]:
        """Extract text while preserving structure and metadata"""
        doc = fitz.open(path)
        structured_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text blocks with position info
            blocks = page.get_text("dict")
            
            page_text = ""
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            # Clean and normalize text
                            text = span["text"].strip()
                            if text:
                                line_text += text + " "
                        
                        if line_text.strip():
                            block_text += line_text.strip() + " "
                    
                    if block_text.strip():
                        page_text += block_text.strip() + "\n"
            
            # Clean up the page text
            page_text = self.clean_text(page_text)
            
            if page_text.strip():
                structured_content.append({
                    'page_num': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
        
        doc.close()
        return structured_content
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = re.sub(r'\n ', '\n', text)  # Remove spaces after newlines
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenated words split across lines
        text = re.sub(r'([.!?])\n([A-Z])', r'\1 \2', text)  # Fix sentence breaks
        
        return text.strip()
    
    def smart_split_text(self, page_data: Dict, chunk_size: int, overlap: int) -> List[Dict]:
        """Split text intelligently preserving sentence boundaries"""
        text = page_data['text']
        page_num = page_data['page_num']
        
        # First split by sentences
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Save current chunk
                chunk_data = {
                    'text': current_chunk.strip(),
                    'page_num': page_num,
                    'sentence_count': len(current_sentences),
                    'char_count': len(current_chunk.strip())
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                current_chunk = " ".join(overlap_sentences)
                current_sentences = overlap_sentences[:]
            
            current_chunk += " " + sentence
            current_sentences.append(sentence)
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'page_num': page_num,
                'sentence_count': len(current_sentences),
                'char_count': len(current_chunk.strip())
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling of edge cases"""
        # Handle common abbreviations that shouldn't trigger sentence breaks
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.', 'vs.', 'e.g.', 'i.e.']
        
        # Temporarily replace abbreviations
        temp_text = text
        for i, abbrev in enumerate(abbreviations):
            temp_text = temp_text.replace(abbrev, f"__ABBREV_{i}__")
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', temp_text)
        
        # Restore abbreviations and clean up
        final_sentences = []
        for sentence in sentences:
            for i, abbrev in enumerate(abbreviations):
                sentence = sentence.replace(f"__ABBREV_{i}__", abbrev)
            
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                final_sentences.append(sentence)
        
        return final_sentences
    
    def create_contextual_chunks(self, structured_content: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Create chunks with enhanced context preservation"""
        all_chunks = []
        text_chunks = []  # For backward compatibility
        
        for page_data in structured_content:
            page_chunks = self.smart_split_text(page_data, CHUNK_SIZE, CHUNK_OVERLAP)
            
            for i, chunk_data in enumerate(page_chunks):
                # Add positional context
                chunk_data['chunk_id'] = len(all_chunks)
                chunk_data['page_chunk_index'] = i
                chunk_data['total_page_chunks'] = len(page_chunks)
                
                # Add neighboring context for better retrieval
                context_text = chunk_data['text']
                
                # Add some context from previous chunk if available
                if i > 0:
                    prev_chunk = page_chunks[i-1]['text']
                    # Take last sentence from previous chunk
                    prev_sentences = self.split_into_sentences(prev_chunk)
                    if prev_sentences:
                        context_text = f"[Previous context: ...{prev_sentences[-1]}] {context_text}"
                
                # Add some context from next chunk if available
                if i < len(page_chunks) - 1:
                    next_chunk = page_chunks[i+1]['text']
                    next_sentences = self.split_into_sentences(next_chunk)
                    if next_sentences:
                        context_text = f"{context_text} [Next context: {next_sentences[0]}...]"
                
                chunk_data['contextual_text'] = context_text
                
                all_chunks.append(chunk_data)
                text_chunks.append(chunk_data['text'])  # Original text for embedding
        
        return all_chunks, text_chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings with progress tracking"""
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(
            chunks, 
            show_progress_bar=True,
            batch_size=32,  # Optimize for memory usage
            convert_to_numpy=True
        )
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build optimized FAISS index"""
        dim = embeddings.shape[1]
        
        # Use IndexHNSWFlat for better performance on larger datasets
        if len(embeddings) > 1000:
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 is M parameter for HNSW
            index.hnsw.efConstruction = 200
        else:
            index = faiss.IndexFlatL2(dim)  # Use flat index for smaller datasets
        
        index.add(embeddings.astype('float32'))
        return index
    
    def save_enhanced_data(self, chunks_metadata: List[Dict], chunks_text: List[str], index: faiss.Index):
        """Save all data with enhanced metadata"""
        # Save FAISS index
        faiss.write_index(index, INDEX_PATH)
        
        # Save chunk text (for backward compatibility)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks_text, f)
        
        # Save enhanced metadata
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(chunks_metadata, f)
        
        # Save processing stats
        stats = {
            'total_chunks': len(chunks_metadata),
            'avg_chunk_size': np.mean([c['char_count'] for c in chunks_metadata]),
            'pages_processed': max([c['page_num'] for c in chunks_metadata]),
            'total_sentences': sum([c['sentence_count'] for c in chunks_metadata])
        }
        
        with open("./company_docs/processing_stats.pkl", "wb") as f:
            pickle.dump(stats, f)
        
        print(f"✅ Saved {len(chunks_metadata)} chunks with enhanced metadata")
        print(f"📊 Stats: {stats}")

def main():
    """Enhanced main pipeline with comprehensive processing"""
    processor = EnhancedPDFProcessor()
    
    print("🔍 Extracting text with structure preservation...")
    structured_content = processor.extract_text_with_structure(PDF_PATH)
    
    print(f"📄 Extracted text from {len(structured_content)} pages")
    
    print("✂️ Creating intelligent chunks...")
    chunks_metadata, chunks_text = processor.create_contextual_chunks(structured_content)
    
    print(f"📦 Created {len(chunks_text)} chunks")
    
    print("🧠 Generating embeddings...")
    embeddings = processor.embed_chunks(chunks_text)
    
    print("🏗️ Building optimized FAISS index...")
    index = processor.build_faiss_index(embeddings)
    
    print("💾 Saving enhanced data...")
    processor.save_enhanced_data(chunks_metadata, chunks_text, index)
    
    print("✅ Enhanced indexing complete with minimal data loss!")

if __name__ == "__main__":
    os.makedirs("./company_docs", exist_ok=True)
    main()