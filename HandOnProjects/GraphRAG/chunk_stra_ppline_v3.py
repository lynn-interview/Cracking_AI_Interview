import os
import re
import json
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass, asdict, field
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    RecursiveCharacterTextSplitter, 
    PythonCodeTextSplitter
)

# --- 1. Data Models ---

@dataclass
class Element:
    type: Literal['Text', 'Table', 'ImageDesc', 'LaTeX', 'Code']
    content: str
    original_index: int

@dataclass
class ProcessedChunk:
    chunk_id: str
    parent_id: str
    content: str
    chunk_type: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

# --- 2. Core Logic Engine ---

class AdvancedContentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.code_splitter = PythonCodeTextSplitter(chunk_size=2000, chunk_overlap=0)

    def preprocess_references(self, text: str) -> str:
        """
        Reference Expansion: Supports [1] and [1, 2] formats.
        """
        # 1. Extract References
        ref_map = {}
        ref_pattern = re.compile(r'(?:^\[(\d+)\]|^(\d+)\.)\s*(.*)$', re.MULTILINE)
        matches = ref_pattern.findall(text)
        for m in matches:
            ref_id = m[0] if m[0] else m[1]
            content = m[2].strip()
            short_desc = content[:50] + "..." if len(content) > 50 else content
            ref_map[ref_id] = short_desc

        if not ref_map:
            return text

        # 2. Replace References in Text
        multi_ref_pattern = re.compile(r'(?<!^)(\[[\d\s,-]+\])')

        def replace_match(match):
            full_str = match.group(1) 
            content_str = full_str[1:-1]
            ids = re.split(r'[,\s]+', content_str)
            
            descriptions = []
            for rid in ids:
                rid = rid.strip()
                if rid in ref_map:
                    descriptions.append(ref_map[rid])
            
            if descriptions:
                joined_desc = "; ".join(descriptions)
                return f"{full_str} (i.e., {joined_desc})"
            
            return full_str

        return multi_ref_pattern.sub(replace_match, text)

    def parse_elements(self, text: str) -> List[Element]:
        """
        Parser: Identifies different components in Markdown/HTML.
        """
        elements = []
        pattern = (
            r'(```python[\s\S]*?```|'  # Code
            r'\$\$[\s\S]*?\$\$|'       # LaTeX $$...$$
            r'\\\[[\s\S]*?\\\]|'       # LaTeX \[...\]
            r'<table>[\s\S]*?</table>|' # HTML Table
            r'(?:^|\n)\|.*\|(?:$|\n))'  # Markdown Table
        )
        
        parts = re.split(pattern, text, flags=re.MULTILINE)
        
        idx = 0
        for part in parts:
            if not part.strip():
                continue
            
            clean_part = part.strip()
            
            if clean_part.startswith("```python"):
                code_content = clean_part.replace("```python", "").replace("```", "").strip()
                elements.append(Element('Code', code_content, idx))
            
            elif clean_part.startswith("$$") or clean_part.startswith("\\["):
                elements.append(Element('LaTeX', clean_part, idx))
                
            elif clean_part.startswith("<table>") or (clean_part.startswith("|") and "\n" in clean_part):
                elements.append(Element('Table', clean_part, idx))
            elif "![" in clean_part:
                elements.append(Element('ImageDesc', clean_part, idx))
            else:
                elements.append(Element('Text', clean_part, idx))
            idx += 1
            
        return elements

    def process_section_elements(self, elements: List[Element]) -> List[Dict]:
        section_chunks = []
        
        for i, el in enumerate(elements):
            
            # --- Strategy A: Caption Binding (Tables/Images) ---
            if el.type in ['Table', 'ImageDesc']:
                combined_content = el.content
                caption_found = False
                
                # Look Back: Check if previous element looks like a caption
                if i > 0 and elements[i-1].type == 'Text':
                    prev_text = elements[i-1].content.strip()
                    lines = prev_text.split('\n')
                    last_line = lines[-1].strip()
                    
                    is_caption = (
                        re.match(r'^(Table|Figure|Fig\.|Chart)\s+\w+', last_line, re.IGNORECASE) or
                        len(last_line) < 100 
                    )
                    
                    if is_caption:
                        combined_content = f"**{last_line}**\n{combined_content}"
                        caption_found = True

                section_chunks.append({
                    "content": combined_content,
                    "type": f"Structure_{el.type}",
                    "meta": {"has_caption": caption_found}
                })
                continue

            # --- Strategy D (NEW): Semantic Chunking for Formulas ---
            # MERGING: Intro + Formula + Explanation
            if el.type == 'LaTeX':
                formula_content = el.content
                
                # 1. Look Behind (Intro Context)
                # Grab the preceding text block to provide context (e.g., "The equation is given by:")
                context_before = ""
                if i > 0 and elements[i-1].type == 'Text':
                    prev_content = elements[i-1].content.strip()
                    # Heuristic: If short, take all. If long, take last ~500 chars (approx 1 paragraph)
                    if len(prev_content) < 500:
                        context_before = prev_content
                    else:
                        context_before = "..." + prev_content[-500:]

                # 2. Look Ahead (Explanation Context)
                # Grab the following text block (e.g., "where alpha is the coefficient...")
                context_after = ""
                if i + 1 < len(elements) and elements[i+1].type == 'Text':
                    next_content = elements[i+1].content.strip()
                    # Heuristic: If short, take all. If long, take first ~500 chars
                    if len(next_content) < 500:
                        context_after = next_content
                    else:
                        context_after = next_content[:500] + "..."

                # 3. Construct Semantic Chunk
                # We merge them into one block. 
                # Note: We do NOT remove the surrounding text blocks from the loop; 
                # they will still be processed as standard text chunks. 
                # This redundancy is intentional ("Semantic Density" for the formula chunk, "Coverage" for text chunks).
                combined_semantic_chunk = f"{context_before}\n\n{formula_content}\n\n{context_after}".strip()
                
                section_chunks.append({
                    "content": combined_semantic_chunk,
                    "type": "Semantic_Formula", 
                    "meta": {"is_formula": True, "semantic_merged": True}
                })
                continue

            # --- Strategy B: Code AST ---
            if el.type == 'Code':
                code_docs = self.code_splitter.create_documents([el.content])
                for doc in code_docs:
                    section_chunks.append({
                        "content": f"```python\n{doc.page_content}\n```",
                        "type": "Code_AST",
                        "meta": {"is_code": True}
                    })
                continue

            # --- Strategy C: Standard Text ---
            if el.type == 'Text':
                text_splits = self.text_splitter.split_text(el.content)
                for txt in text_splits:
                    section_chunks.append({
                        "content": txt,
                        "type": "Standard_Text",
                        "meta": {}
                    })
        
        return section_chunks

# --- 3. Pipeline Engineering ---

class GraphReadyPipeline:
    def __init__(self, output_dir="graph_data_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.processor = AdvancedContentProcessor()
        self.parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2")]
        )

    def run(self, file_path: str):
        print(f"🚀 Starting processing: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 1. Pre-processing
        clean_text = self.processor.preprocess_references(raw_text)
        clean_text = re.sub(r'^##\s+(Page\s+\d+.*)$', r'\n\n**\1**\n\n', clean_text, flags=re.MULTILINE)
        
        # 2. Parent Splitting
        parent_docs = self.parent_splitter.split_text(clean_text)
        
        final_chunks: List[ProcessedChunk] = []
        
        # 3. Element Processing
        for i, parent in enumerate(parent_docs):
            parent_id = f"P_{i:03d}"
            
            # Parse
            elements = self.processor.parse_elements(parent.page_content)
            
            # Process (with Semantic Merging)
            processed_sub_chunks = self.processor.process_section_elements(elements)
            
            for j, item in enumerate(processed_sub_chunks):
                chunk_id = f"{parent_id}_C_{j:03d}"
                
                combined_meta = parent.metadata.copy()
                combined_meta.update(item['meta'])
                combined_meta.update({"source_file": os.path.basename(file_path)})
                
                final_chunks.append(ProcessedChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    content=item['content'],
                    chunk_type=item['type'],
                    metadata=combined_meta
                ))

        self._save_results(final_chunks)
        return final_chunks

    def _save_results(self, chunks: List[ProcessedChunk]):
        # Save JSONL
        jsonl_path = os.path.join(self.output_dir, "graph_source_v3.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        
        # Save Markdown Debug Report
        md_path = os.path.join(self.output_dir, "human_check_final_v3.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Chunking Strategy Debug Report\n\n")
            curr_parent = None
            for c in chunks:
                if c.parent_id != curr_parent:
                    f.write(f"\n## Parent Section: {c.parent_id}\nMetaData: {c.metadata}\n\n")
                    curr_parent = c.parent_id
                
                f.write(f"#### Child Chunk: {c.chunk_id} ({c.chunk_type})\n")
                f.write(f"> {c.content.replace(chr(10), chr(10)+'> ')}\n\n")
                f.write("---\n")

        print(f"✅ Processing Complete!\nData saved to: {self.output_dir}")

# --- 4. Execution ---

if __name__ == "__main__":  
    import sys

    # Default test content if no file provided
    if len(sys.argv) < 2:
        test_content = """
# Physics of Ultrasound

The propagation of sound waves in a medium is described by the wave equation. A specific solution for plane waves is:

$$p(x,t) = A e^{i(kx - \omega t)}$$

Where $A$ is the amplitude, $k$ is the wavenumber, and $\omega$ is the angular frequency. This equation assumes a lossless medium.

However, in tissue, we must account for attenuation.
"""
        with open("test_physics.md", "w", encoding="utf-8") as f:
            f.write(test_content)
        input_filepath = "test_physics.md"    
    else:
        input_filepath = sys.argv[1]
        
    pipeline = GraphReadyPipeline()
    pipeline.run(input_filepath)