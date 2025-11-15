#!/usr/bin/env python3
"""
Comprehensive Repository Documentation Generator

This script generates exhaustive documentation for the entire repository,
creating a complete reference system with:
- Per-file documentation (_docs.md) and keyword indexes (_kw.md)
- Per-folder indexes (index.md), documentation (doc.md), and subtree keywords (sub.md)
- Global comprehensive book, keyword index, and root index
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class RepoDocumentationGenerator:
    def __init__(self, repo_root: str, docs_dir: str = "docs"):
        self.repo_root = Path(repo_root).resolve()
        self.docs_dir = self.repo_root / docs_dir
        self.file_tree = {}
        self.all_keywords = defaultdict(list)
        self.folder_keywords = defaultdict(lambda: defaultdict(list))

        # Track all files and folders
        self.all_files = []
        self.all_folders = []

    def scan_repository(self):
        """Scan the repository and build a complete file tree."""
        print("Scanning repository structure...")

        for root, dirs, files in os.walk(self.repo_root):
            # Skip .git, docs, and other hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'docs']

            root_path = Path(root)
            rel_path = root_path.relative_to(self.repo_root)

            if rel_path != Path('.'):
                self.all_folders.append(rel_path)

            for file in files:
                if not file.startswith('.'):
                    file_path = root_path / file
                    rel_file_path = file_path.relative_to(self.repo_root)
                    self.all_files.append(rel_file_path)

        print(f"Found {len(self.all_files)} files in {len(self.all_folders) + 1} folders")

    def read_file_content(self, file_path: Path) -> str:
        """Read and return file content, handling different encodings."""
        try:
            with open(self.repo_root / file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(self.repo_root / file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except:
                return "[Binary file - content not readable as text]"
        except Exception as e:
            return f"[Error reading file: {e}]"

    def extract_keywords_from_content(self, content: str, file_path: Path) -> List[Tuple[str, str]]:
        """Extract keywords and their descriptions from file content."""
        keywords = []

        # Extract Python/code identifiers
        if file_path.suffix in ['.py', '.js', '.ts', '.cpp', '.h', '.c']:
            # Classes
            class_matches = re.findall(r'class\s+(\w+)', content)
            for cls in class_matches:
                keywords.append((cls, f"Class definition: {cls}"))

            # Functions/methods
            func_matches = re.findall(r'def\s+(\w+)', content)
            for func in func_matches:
                keywords.append((func, f"Function/method: {func}"))

            # Import statements
            import_matches = re.findall(r'import\s+([\w.]+)', content)
            for imp in import_matches:
                keywords.append((imp, f"Imported module: {imp}"))

            # from X import Y
            from_matches = re.findall(r'from\s+([\w.]+)\s+import', content)
            for imp in from_matches:
                keywords.append((imp, f"Imported from: {imp}"))

        # Extract markdown headers
        if file_path.suffix == '.md':
            header_matches = re.findall(r'#+\s+(.+)', content)
            for header in header_matches:
                clean_header = header.strip()
                keywords.append((clean_header, f"Section: {clean_header}"))

        # Extract common technical terms
        technical_terms = [
            'tensorflow', 'neural network', 'lstm', 'rnn', 'cnn', 'gan', 'autoencoder',
            'regression', 'classification', 'optimizer', 'gradient', 'loss', 'accuracy',
            'training', 'testing', 'validation', 'dataset', 'mnist', 'cifar',
            'convolution', 'pooling', 'activation', 'relu', 'sigmoid', 'softmax',
            'backpropagation', 'forward pass', 'embedding', 'attention', 'transformer'
        ]

        content_lower = content.lower()
        for term in technical_terms:
            if term in content_lower:
                keywords.append((term, f"Technical term: {term}"))

        return list(set(keywords))  # Remove duplicates

    def generate_file_docs(self, file_path: Path) -> str:
        """Generate comprehensive documentation for a single file."""
        content = self.read_file_content(file_path)
        file_size = (self.repo_root / file_path).stat().st_size

        # Extract detailed information
        keywords = self.extract_keywords_from_content(content, file_path)

        # Store keywords for global index
        for kw, desc in keywords:
            self.all_keywords[kw].append(file_path)

        # Determine file purpose
        purpose = self.determine_file_purpose(file_path, content)

        # Prepare content display (handle long content)
        truncation_note = "\n... [Content truncated for brevity] ..."
        display_content = content[:10000] if len(content) <= 10000 else content[:10000] + truncation_note

        doc = f"""# Documentation: {file_path}

## File Metadata

- **File Path**: `{file_path}`
- **File Size**: {file_size} bytes
- **File Type**: {file_path.suffix or 'No extension'}
- **Purpose**: {purpose}

## Original Source

```{self.get_code_fence_language(file_path)}
{display_content}
```

## High-Level Overview

This file is located at `{file_path}` within the TensorFlow Examples repository.

{self.generate_high_level_overview(file_path, content)}

## Detailed Walkthrough

{self.generate_detailed_walkthrough(file_path, content)}

## Inline Code Examples

{self.generate_code_examples(file_path, content)}

## Design & Architecture

{self.generate_architecture_section(file_path, content)}

## Performance & Complexity

{self.generate_performance_analysis(file_path, content)}

## Security & Safety Considerations

{self.generate_security_section(file_path, content)}

## Alternatives & Variants

{self.generate_alternatives_section(file_path, content)}

## Testing & Usage Notes

{self.generate_testing_section(file_path, content)}

## Related Files

{self.generate_related_files_section(file_path)}

## Keywords

{', '.join([kw for kw, _ in keywords])}

---

*This documentation was automatically generated for comprehensive repository understanding.*
"""
        return doc

    def generate_file_keywords(self, file_path: Path) -> str:
        """Generate keyword index for a single file."""
        content = self.read_file_content(file_path)
        keywords = self.extract_keywords_from_content(content, file_path)

        # Calculate relative paths
        docs_file_path = self.docs_dir / file_path.parent / f"{file_path.name}_docs.md"
        docs_rel_from_kw = os.path.relpath(docs_file_path, self.docs_dir / file_path.parent)
        source_rel_from_kw = os.path.relpath(self.repo_root / file_path, self.docs_dir / file_path.parent)

        kw_doc = f"""# Keyword Map: {file_path}

## File Path and Links

- **Original Source**: [{file_path}]({source_rel_from_kw})
- **Documentation**: [{file_path}_docs.md]({docs_rel_from_kw})

## Keywords

This file contains the following key concepts, terms, and identifiers:

"""

        if keywords:
            for kw, desc in sorted(keywords, key=lambda x: x[0].lower()):
                kw_doc += f"### {kw}\n\n"
                kw_doc += f"- **Description**: {desc}\n"
                kw_doc += f"- **File**: [{file_path}]({source_rel_from_kw})\n"
                kw_doc += f"- **Documentation**: [{file_path}_docs.md]({docs_rel_from_kw})\n\n"
        else:
            kw_doc += "*No keywords extracted from this file.*\n\n"

        kw_doc += """
## Keyword → Section Map

The keywords above can be found in the following sections of the documentation:

- See the main [documentation file]({docs_rel_from_kw}) for detailed coverage of all keywords.

---

*This keyword index was automatically generated for comprehensive repository navigation.*
"""
        return kw_doc

    def generate_folder_index(self, folder_path: Path) -> str:
        """Generate index.md for a folder."""
        # Get immediate children
        subfolders = []
        files = []

        for item in self.all_folders:
            if item.parent == folder_path:
                subfolders.append(item)

        for item in self.all_files:
            if item.parent == folder_path:
                files.append(item)

        # Root folder special case
        if folder_path == Path('.'):
            folder_display = "Root Directory"
            folder_path_str = "."
        else:
            folder_display = f"`{folder_path}/`"
            folder_path_str = str(folder_path)

        index_doc = f"""# Index: {folder_display}

## Overview

This folder contains the following content within the TensorFlow Examples repository.

{self.generate_folder_overview(folder_path)}

## Subfolders

"""
        if subfolders:
            for subfolder in sorted(subfolders):
                subfolder_name = subfolder.name
                rel_path = f"./{subfolder_name}/index.md"
                index_doc += f"- **[{subfolder_name}/]({rel_path})**: {self.get_folder_description(subfolder)}\n"
        else:
            index_doc += "*No subfolders in this directory.*\n"

        index_doc += "\n## Files\n\n"

        if files:
            index_doc += "| Filename | Description | Documentation | Keywords |\n"
            index_doc += "|----------|-------------|---------------|----------|\n"

            for file in sorted(files):
                filename = file.name
                desc = self.get_file_description(file)
                docs_link = f"./{filename}_docs.md"
                kw_link = f"./{filename}_kw.md"
                source_link = os.path.relpath(self.repo_root / file, self.docs_dir / folder_path)

                index_doc += f"| [{filename}]({source_link}) | {desc} | [docs]({docs_link}) | [keywords]({kw_link}) |\n"
        else:
            index_doc += "*No files in this directory.*\n"

        index_doc += """
## Navigation Hints

"""
        index_doc += self.generate_navigation_hints(folder_path)

        index_doc += """
---

*This index was automatically generated for comprehensive repository navigation.*
"""
        return index_doc

    def generate_folder_doc(self, folder_path: Path) -> str:
        """Generate doc.md narrative documentation for a folder."""
        folder_display = str(folder_path) if folder_path != Path('.') else "Root Directory"

        doc = f"""# Documentation: {folder_display}

## Role in the Project

{self.generate_folder_role(folder_path)}

## Key Concepts

{self.generate_folder_concepts(folder_path)}

## Important Files

{self.generate_important_files_list(folder_path)}

## Data Flows & Interactions

{self.generate_data_flows(folder_path)}

## How to Work with This Folder

{self.generate_workflow_guidance(folder_path)}

## Cross References

{self.generate_cross_references(folder_path)}

---

*This folder documentation was automatically generated as part of the comprehensive repository book.*
"""
        return doc

    def generate_folder_subtree_keywords(self, folder_path: Path) -> str:
        """Generate sub.md keyword index for a folder and all descendants."""
        # Collect all files in this folder and subfolders
        descendant_files = []

        for file in self.all_files:
            if folder_path == Path('.'):
                descendant_files.append(file)
            elif str(file).startswith(str(folder_path) + os.sep):
                descendant_files.append(file)
            elif file.parent == folder_path:
                descendant_files.append(file)

        # Collect all keywords from descendants
        subtree_keywords = defaultdict(list)

        for file in descendant_files:
            content = self.read_file_content(file)
            keywords = self.extract_keywords_from_content(content, file)
            for kw, desc in keywords:
                subtree_keywords[kw].append((file, desc))

        folder_display = str(folder_path) if folder_path != Path('.') else "Root Directory"

        sub_doc = f"""# Subtree Keyword Index: {folder_display}

## Scope

This keyword index covers all files within `{folder_path}/` and all its subdirectories recursively.

Total files indexed: {len(descendant_files)}

## Keywords A–Z

"""

        if subtree_keywords:
            for kw in sorted(subtree_keywords.keys(), key=str.lower):
                sub_doc += f"### {kw}\n\n"
                sub_doc += "**Found in:**\n\n"

                for file, desc in sorted(subtree_keywords[kw], key=lambda x: str(x[0])):
                    # Calculate relative paths
                    docs_path = os.path.relpath(
                        self.docs_dir / file.parent / f"{file.name}_docs.md",
                        self.docs_dir / folder_path
                    )
                    kw_path = os.path.relpath(
                        self.docs_dir / file.parent / f"{file.name}_kw.md",
                        self.docs_dir / folder_path
                    )
                    source_path = os.path.relpath(
                        self.repo_root / file,
                        self.docs_dir / folder_path
                    )

                    sub_doc += f"- `{file}`: {desc}\n"
                    sub_doc += f"  - [Documentation]({docs_path})\n"
                    sub_doc += f"  - [Keywords]({kw_path})\n"
                    sub_doc += f"  - [Source]({source_path})\n\n"
        else:
            sub_doc += "*No keywords found in this subtree.*\n\n"

        sub_doc += """
## Folder-Level Navigation

Browse by topic area within this subtree:

"""
        sub_doc += self.generate_topic_navigation(folder_path, subtree_keywords)

        sub_doc += """
---

*This subtree keyword index was automatically generated for comprehensive repository navigation.*
"""
        return sub_doc

    # Helper methods for content generation

    def get_code_fence_language(self, file_path: Path) -> str:
        """Get the appropriate code fence language for syntax highlighting."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.java': 'java',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.sh': 'bash',
            '.txt': 'text'
        }
        return ext_map.get(file_path.suffix, '')

    def determine_file_purpose(self, file_path: Path, content: str) -> str:
        """Determine the purpose of a file based on its path and content."""
        path_str = str(file_path).lower()

        if 'readme' in path_str:
            return "Repository documentation and guide"
        elif 'license' in path_str:
            return "Software license file"
        elif '.ipynb' in path_str:
            return "Jupyter notebook with interactive code examples"
        elif 'test' in path_str:
            return "Test file for code validation"
        elif 'introduction' in path_str:
            return "Introductory tutorial or example"
        elif 'basicmodels' in path_str or 'basic_models' in path_str:
            return "Basic machine learning model implementation"
        elif 'neuralnetworks' in path_str or 'neural_networks' in path_str:
            return "Neural network implementation or example"
        elif 'utils' in path_str:
            return "Utility functions and helper code"
        elif 'datamanagement' in path_str or 'data_management' in path_str:
            return "Data loading, processing, and management utilities"
        elif 'multigpu' in path_str or 'multi_gpu' in path_str:
            return "Multi-GPU training and parallel processing example"
        elif file_path.suffix == '.py':
            return "Python source code file"
        elif file_path.suffix == '.md':
            return "Markdown documentation file"
        elif file_path.suffix in ['.png', '.jpg', '.jpeg', '.gif']:
            return "Image resource file"
        else:
            return "Repository file"

    def generate_high_level_overview(self, file_path: Path, content: str) -> str:
        """Generate high-level overview based on file analysis."""
        path_parts = file_path.parts

        overview = f"### Purpose\n\nThis file "

        if file_path.suffix == '.py':
            if 'class ' in content:
                overview += "implements one or more Python classes "
            if 'def ' in content:
                overview += "defines functions and methods "
            overview += "as part of the TensorFlow Examples repository.\n\n"
        elif file_path.suffix == '.ipynb':
            overview += "is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.\n\n"
        elif file_path.suffix == '.md':
            overview += "provides documentation in Markdown format.\n\n"
        else:
            overview += f"serves as a {self.determine_file_purpose(file_path, content).lower()}.\n\n"

        overview += "### Context\n\n"

        if len(path_parts) > 1:
            overview += f"Located in the `{path_parts[0]}/` directory"
            if len(path_parts) > 2:
                overview += f", specifically within `{'/'.join(path_parts[:-1])}/`"
            overview += ", this file is part of "

            if 'tensorflow_v1' in str(file_path):
                overview += "the TensorFlow 1.x examples collection"
            elif 'tensorflow_v2' in str(file_path):
                overview += "the TensorFlow 2.x examples collection"
            elif 'examples' in str(file_path):
                overview += "the main examples collection (defaulting to latest TensorFlow)"
            elif 'notebooks' in str(file_path):
                overview += "the notebook-based tutorials"
            else:
                overview += "the repository infrastructure"

            overview += ".\n\n"

        return overview

    def generate_detailed_walkthrough(self, file_path: Path, content: str) -> str:
        """Generate detailed walkthrough of file contents."""
        walkthrough = ""

        if file_path.suffix == '.py':
            # Analyze Python code structure
            lines = content.split('\n')

            # Find imports
            imports = [line for line in lines if line.strip().startswith(('import ', 'from '))]
            if imports:
                walkthrough += "### Imports and Dependencies\n\n"
                walkthrough += "This file imports the following modules:\n\n"
                for imp in imports[:20]:  # Limit to first 20 imports
                    walkthrough += f"- `{imp.strip()}`\n"
                walkthrough += "\n"

            # Find class definitions
            classes = re.findall(r'class\s+(\w+).*?:', content)
            if classes:
                walkthrough += "### Class Definitions\n\n"
                for cls in classes:
                    walkthrough += f"#### {cls}\n\n"
                    walkthrough += f"This class provides functionality for {cls.lower()} operations.\n\n"

            # Find function definitions
            functions = re.findall(r'def\s+(\w+)\s*\((.*?)\)', content)
            if functions:
                walkthrough += "### Functions and Methods\n\n"
                for func_name, params in functions[:15]:  # Limit to first 15 functions
                    walkthrough += f"#### `{func_name}({params})`\n\n"
                    walkthrough += f"This function handles {func_name.replace('_', ' ')} functionality.\n\n"

        elif file_path.suffix == '.md':
            # Analyze markdown structure
            headers = re.findall(r'(#+)\s+(.+)', content)
            if headers:
                walkthrough += "### Document Structure\n\n"
                walkthrough += "This document contains the following sections:\n\n"
                for level, title in headers[:30]:
                    indent = "  " * (len(level) - 1)
                    walkthrough += f"{indent}- {title}\n"
                walkthrough += "\n"

        elif file_path.suffix == '.ipynb':
            walkthrough += "### Notebook Structure\n\n"
            walkthrough += "This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.\n\n"

            try:
                notebook_data = json.loads(content)
                if 'cells' in notebook_data:
                    cell_count = len(notebook_data['cells'])
                    code_cells = sum(1 for cell in notebook_data['cells'] if cell.get('cell_type') == 'code')
                    markdown_cells = sum(1 for cell in notebook_data['cells'] if cell.get('cell_type') == 'markdown')

                    walkthrough += f"- Total cells: {cell_count}\n"
                    walkthrough += f"- Code cells: {code_cells}\n"
                    walkthrough += f"- Markdown cells: {markdown_cells}\n\n"
            except:
                pass

        if not walkthrough:
            walkthrough = "This file contains content specific to its purpose in the repository. See the original source above for complete details.\n\n"

        return walkthrough

    def generate_code_examples(self, file_path: Path, content: str) -> str:
        """Generate code examples section."""
        examples = "### Example Usage\n\n"

        if file_path.suffix == '.py':
            # Check if there's a main block
            if 'if __name__' in content:
                examples += "This file can be run directly as a script:\n\n"
                examples += f"```bash\npython {file_path}\n```\n\n"

            examples += "The code in this file demonstrates TensorFlow patterns and can be used as a reference for implementing similar functionality.\n\n"

        elif file_path.suffix == '.ipynb':
            examples += "Open this notebook in Jupyter:\n\n"
            examples += f"```bash\njupyter notebook {file_path}\n```\n\n"
            examples += "Then execute cells sequentially to see TensorFlow in action.\n\n"

        return examples

    def generate_architecture_section(self, file_path: Path, content: str) -> str:
        """Generate architecture and design section."""
        arch = "### Architectural Context\n\n"

        path_parts = list(file_path.parts)

        if 'examples' in path_parts or 'notebooks' in path_parts:
            arch += "This file is part of the TensorFlow Examples educational repository structure. "

            if 'Introduction' in str(file_path) or '1_' in str(file_path):
                arch += "It belongs to the introductory section, designed for beginners.\n\n"
            elif 'BasicModels' in str(file_path) or '2_' in str(file_path):
                arch += "It demonstrates basic machine learning models.\n\n"
            elif 'NeuralNetworks' in str(file_path) or '3_' in str(file_path):
                arch += "It implements neural network architectures and techniques.\n\n"
            elif 'Utils' in str(file_path) or '4_' in str(file_path):
                arch += "It provides utility functions for TensorFlow operations.\n\n"
            elif 'DataManagement' in str(file_path) or '5_' in str(file_path):
                arch += "It demonstrates data loading and preprocessing techniques.\n\n"
            elif 'MultiGPU' in str(file_path) or '6_' in str(file_path):
                arch += "It shows how to leverage multiple GPUs for training.\n\n"

        arch += "### Design Patterns\n\n"

        if 'tensorflow' in content.lower():
            arch += "- Uses TensorFlow framework for machine learning operations\n"
        if 'class ' in content:
            arch += "- Implements object-oriented design with custom classes\n"
        if 'def ' in content and file_path.suffix == '.py':
            arch += "- Organizes functionality into modular functions\n"

        return arch

    def generate_performance_analysis(self, file_path: Path, content: str) -> str:
        """Generate performance and complexity analysis."""
        perf = "### Performance Characteristics\n\n"

        if 'gpu' in str(file_path).lower() or 'multigpu' in content.lower():
            perf += "- **GPU Acceleration**: This code is designed to leverage GPU hardware for accelerated computation\n"
            perf += "- **Scalability**: Implements multi-GPU training for handling large-scale models and datasets\n"

        if any(term in content.lower() for term in ['batch', 'epoch', 'iteration']):
            perf += "- **Batch Processing**: Uses batched operations for efficient data processing\n"
            perf += "- **Training Optimization**: Implements iterative training with multiple epochs\n"

        if any(term in content.lower() for term in ['cnn', 'convolutional', 'lstm', 'rnn']):
            perf += "- **Computational Complexity**: Neural network operations can be computationally intensive\n"
            perf += "- **Memory Usage**: Deep learning models require significant memory for parameters and activations\n"

        perf += "\nFor optimal performance, ensure appropriate hardware resources and TensorFlow GPU support if applicable.\n\n"

        return perf

    def generate_security_section(self, file_path: Path, content: str) -> str:
        """Generate security and safety considerations."""
        security = "### Security Considerations\n\n"

        security += "As educational example code:\n\n"
        security += "- **Input Validation**: Production use should add input validation and sanitization\n"
        security += "- **Data Privacy**: Be cautious when training on sensitive data\n"
        security += "- **Model Security**: Trained models can potentially leak information about training data\n"

        if 'load' in content.lower() or 'download' in content.lower():
            security += "- **Data Sources**: Verify integrity of downloaded datasets and models\n"

        if 'save' in content.lower() or 'checkpoint' in content.lower():
            security += "- **File Permissions**: Ensure saved models and checkpoints have appropriate access controls\n"

        security += "\n"
        return security

    def generate_alternatives_section(self, file_path: Path, content: str) -> str:
        """Generate alternatives and variants section."""
        alt = "### Alternative Approaches\n\n"

        if 'eager' in str(file_path).lower():
            alt += "- **Graph Mode**: This uses eager execution; graph mode (TF 1.x style) is an alternative\n"
        elif 'raw' in str(file_path).lower():
            alt += "- **High-Level APIs**: This uses low-level operations; `tf.keras` provides higher-level abstractions\n"
        else:
            alt += "- **Different Frameworks**: PyTorch, JAX, or MXNet could be used for similar functionality\n"

        alt += "- **Model Architectures**: Various network architectures can solve similar problems\n"
        alt += "- **Training Strategies**: Different optimizers, learning rates, and regularization techniques are possible\n\n"

        return alt

    def generate_testing_section(self, file_path: Path, content: str) -> str:
        """Generate testing and usage notes."""
        testing = "### Testing Recommendations\n\n"

        if file_path.suffix == '.py':
            testing += f"To test this module:\n\n"
            testing += f"```bash\npython {file_path}\n```\n\n"
            testing += "Verify that:\n"
            testing += "- TensorFlow is properly installed\n"
            testing += "- Required datasets are accessible\n"
            testing += "- Output matches expected results\n\n"

        elif file_path.suffix == '.ipynb':
            testing += "To test this notebook:\n\n"
            testing += "```bash\njupyter notebook\n```\n\n"
            testing += "Then:\n"
            testing += "1. Open the notebook\n"
            testing += "2. Run all cells sequentially\n"
            testing += "3. Verify outputs and visualizations\n\n"

        testing += "### Usage Notes\n\n"
        testing += "- Ensure TensorFlow is installed: `pip install tensorflow`\n"
        testing += "- Some examples may require additional dependencies\n"
        testing += "- GPU support is optional but recommended for large models\n\n"

        return testing

    def generate_related_files_section(self, file_path: Path) -> str:
        """Generate related files section."""
        related = "### Related Files in Repository\n\n"

        # Find files in same directory
        same_dir_files = [f for f in self.all_files if f.parent == file_path.parent and f != file_path]

        if same_dir_files:
            related += "Files in the same directory:\n\n"
            for rel_file in sorted(same_dir_files)[:10]:
                rel_path = os.path.relpath(
                    self.docs_dir / rel_file.parent / f"{rel_file.name}_docs.md",
                    self.docs_dir / file_path.parent
                )
                related += f"- [{rel_file.name}]({rel_path})\n"
            related += "\n"

        # Find related files by topic
        file_stem = file_path.stem.lower()

        if '.ipynb' in str(file_path):
            # Look for corresponding .py file
            py_equiv = file_path.with_suffix('.py')
            alt_path = Path(str(file_path).replace('notebooks', 'examples'))

            related += "Related implementations:\n\n"
            related += f"- Check `examples/` directory for Python script versions\n"

        related += "\nSee the [folder index](./index.md) for a complete list of related files.\n\n"

        return related

    def generate_folder_overview(self, folder_path: Path) -> str:
        """Generate overview for a folder."""
        path_str = str(folder_path).lower()

        if folder_path == Path('.'):
            return "The root directory of the TensorFlow Examples repository, containing examples and tutorials for learning TensorFlow."
        elif 'introduction' in path_str or '1_' in path_str:
            return "Introductory examples for getting started with TensorFlow."
        elif 'basicmodels' in path_str or '2_' in path_str:
            return "Basic machine learning models including regression, classification, and clustering."
        elif 'neuralnetworks' in path_str or '3_' in path_str:
            return "Neural network implementations including CNNs, RNNs, GANs, and autoencoders."
        elif 'utils' in path_str or '4_' in path_str:
            return "Utility code for model saving, TensorBoard integration, and other helper functionality."
        elif 'datamanagement' in path_str or '5_' in path_str:
            return "Data loading, preprocessing, and augmentation examples."
        elif 'multigpu' in path_str or '6_' in path_str or 'hardware' in path_str:
            return "Examples for multi-GPU training and hardware optimization."
        elif 'tensorflow_v1' in path_str:
            return "TensorFlow 1.x specific examples and code."
        elif 'tensorflow_v2' in path_str:
            return "TensorFlow 2.x specific examples and code."
        elif 'notebooks' in path_str:
            return "Jupyter notebook versions of the examples."
        elif 'examples' in path_str:
            return "Python script versions of the examples."
        elif 'resources' in path_str:
            return "Resource files including images and assets."
        else:
            return f"Content within the {folder_path} directory."

    def get_folder_description(self, folder_path: Path) -> str:
        """Get brief description of a folder."""
        return self.generate_folder_overview(folder_path)

    def get_file_description(self, file_path: Path) -> str:
        """Get brief description of a file."""
        content = self.read_file_content(file_path)
        return self.determine_file_purpose(file_path, content)

    def generate_navigation_hints(self, folder_path: Path) -> str:
        """Generate navigation hints for a folder."""
        hints = ""

        if folder_path == Path('.'):
            hints += "Start with:\n\n"
            hints += "1. Read [README.md](../README.md) for repository overview\n"
            hints += "2. Browse subfolders by topic area\n"
            hints += "3. Check [comprehensive_book.md](./comprehensive_book.md) for complete documentation\n\n"
        else:
            hints += "To navigate this folder:\n\n"
            hints += "1. Review this index to understand available files\n"
            hints += "2. Read [doc.md](./doc.md) for folder-level documentation\n"
            hints += "3. Check [sub.md](./sub.md) for keyword-based navigation\n"
            hints += "4. Dive into individual file documentation as needed\n\n"

        return hints

    def generate_folder_role(self, folder_path: Path) -> str:
        """Generate detailed role description for folder."""
        role = self.generate_folder_overview(folder_path)
        role += "\n\n"

        if 'examples' in str(folder_path):
            role += "This folder contains executable Python scripts that demonstrate TensorFlow concepts. "
            role += "Each file can be run independently to see the model in action."
        elif 'notebooks' in str(folder_path):
            role += "This folder contains Jupyter notebooks with interactive code cells, explanatory text, and visualizations. "
            role += "These provide an excellent learning environment for understanding TensorFlow step-by-step."

        role += "\n"
        return role

    def generate_folder_concepts(self, folder_path: Path) -> str:
        """Generate key concepts for folder."""
        concepts = ""
        path_str = str(folder_path).lower()

        if 'introduction' in path_str:
            concepts += "- TensorFlow basics\n"
            concepts += "- Tensor operations\n"
            concepts += "- Computational graphs\n"
            concepts += "- Eager execution\n"
        elif 'basicmodels' in path_str:
            concepts += "- Supervised learning\n"
            concepts += "- Linear regression\n"
            concepts += "- Logistic regression\n"
            concepts += "- Decision trees\n"
            concepts += "- K-means clustering\n"
        elif 'neuralnetworks' in path_str:
            concepts += "- Artificial neural networks\n"
            concepts += "- Convolutional neural networks (CNNs)\n"
            concepts += "- Recurrent neural networks (RNNs/LSTMs)\n"
            concepts += "- Generative adversarial networks (GANs)\n"
            concepts += "- Autoencoders\n"
            concepts += "- Backpropagation\n"
        elif 'utils' in path_str:
            concepts += "- Model persistence (saving/loading)\n"
            concepts += "- TensorBoard visualization\n"
            concepts += "- Custom layers and modules\n"
        elif 'datamanagement' in path_str:
            concepts += "- Data pipeline optimization\n"
            concepts += "- TFRecord format\n"
            concepts += "- Image augmentation\n"
            concepts += "- Dataset API\n"
        elif 'multigpu' in path_str or 'hardware' in path_str:
            concepts += "- Distributed training\n"
            concepts += "- Multi-GPU strategies\n"
            concepts += "- Hardware acceleration\n"

        if not concepts:
            concepts = "Concepts specific to this folder's purpose.\n"

        return concepts + "\n"

    def generate_important_files_list(self, folder_path: Path) -> str:
        """Generate list of important files in folder."""
        files_in_folder = [f for f in self.all_files if f.parent == folder_path]

        if not files_in_folder:
            return "*No files in this folder.*\n\n"

        important = ""

        # Prioritize certain file types
        readmes = [f for f in files_in_folder if 'readme' in f.name.lower()]
        py_files = [f for f in files_in_folder if f.suffix == '.py']
        notebooks = [f for f in files_in_folder if f.suffix == '.ipynb']

        if readmes:
            important += "### Documentation\n\n"
            for f in readmes:
                doc_link = f"./{f.name}_docs.md"
                important += f"- [{f.name}]({doc_link}): Overview and guide for this folder\n"
            important += "\n"

        if notebooks:
            important += "### Jupyter Notebooks\n\n"
            for f in sorted(notebooks)[:5]:
                doc_link = f"./{f.name}_docs.md"
                important += f"- [{f.name}]({doc_link}): {self.get_file_description(f)}\n"
            if len(notebooks) > 5:
                important += f"\n... and {len(notebooks) - 5} more notebooks\n"
            important += "\n"

        if py_files:
            important += "### Python Scripts\n\n"
            for f in sorted(py_files)[:5]:
                doc_link = f"./{f.name}_docs.md"
                important += f"- [{f.name}]({doc_link}): {self.get_file_description(f)}\n"
            if len(py_files) > 5:
                important += f"\n... and {len(py_files) - 5} more scripts\n"
            important += "\n"

        return important

    def generate_data_flows(self, folder_path: Path) -> str:
        """Generate data flow and interaction description."""
        flows = "### Typical Workflow\n\n"

        path_str = str(folder_path).lower()

        if 'introduction' in path_str:
            flows += "1. Start with basic operations to understand TensorFlow tensors\n"
            flows += "2. Learn about computational graphs and execution models\n"
            flows += "3. Explore eager execution for interactive development\n\n"
        elif any(x in path_str for x in ['basicmodels', 'neuralnetworks']):
            flows += "1. Load and prepare dataset (often MNIST or similar)\n"
            flows += "2. Define model architecture and parameters\n"
            flows += "3. Set up training loop with optimizer and loss function\n"
            flows += "4. Train model on training data\n"
            flows += "5. Evaluate model on test data\n"
            flows += "6. Visualize results and metrics\n\n"
        elif 'datamanagement' in path_str:
            flows += "1. Load raw data from source (files, arrays, etc.)\n"
            flows += "2. Apply preprocessing and augmentation\n"
            flows += "3. Create efficient data pipeline\n"
            flows += "4. Feed batches to model during training\n\n"
        elif 'utils' in path_str:
            flows += "1. Use utilities during model development\n"
            flows += "2. Save models after training\n"
            flows += "3. Load models for inference or continued training\n"
            flows += "4. Monitor training with TensorBoard\n\n"

        flows += "### Interactions with Other Components\n\n"
        flows += "Files in this folder interact with:\n\n"
        flows += "- TensorFlow core framework\n"
        flows += "- Dataset utilities for data loading\n"
        flows += "- Other examples as reference implementations\n\n"

        return flows

    def generate_workflow_guidance(self, folder_path: Path) -> str:
        """Generate guidance for working with folder."""
        guidance = "### Getting Started\n\n"

        guidance += "To work with files in this folder:\n\n"
        guidance += "1. Ensure TensorFlow is installed: `pip install tensorflow`\n"
        guidance += "2. Review the documentation for individual files\n"
        guidance += "3. Start with simpler examples before complex ones\n"
        guidance += "4. Experiment by modifying parameters and observing results\n\n"

        guidance += "### Extending This Code\n\n"
        guidance += "To extend or modify:\n\n"
        guidance += "- Use these examples as templates for your own models\n"
        guidance += "- Substitute your own datasets\n"
        guidance += "- Adjust hyperparameters for different results\n"
        guidance += "- Combine techniques from multiple examples\n\n"

        guidance += "### Best Practices\n\n"
        guidance += "- Keep track of experiments and results\n"
        guidance += "- Use version control for your modifications\n"
        guidance += "- Document your changes and reasoning\n"
        guidance += "- Share improvements with the community\n\n"

        return guidance

    def generate_cross_references(self, folder_path: Path) -> str:
        """Generate cross-references to related folders."""
        refs = "### Related Folders\n\n"

        # Find sibling folders
        if folder_path.parent == Path('.'):
            siblings = [f for f in self.all_folders if f.parent == Path('.')]
        else:
            siblings = [f for f in self.all_folders if f.parent == folder_path.parent and f != folder_path]

        if siblings:
            refs += "Sibling folders:\n\n"
            for sib in sorted(siblings)[:5]:
                rel_path = os.path.relpath(
                    self.docs_dir / sib / "doc.md",
                    self.docs_dir / folder_path
                )
                refs += f"- [{sib}]({rel_path}): {self.get_folder_description(sib)}\n"
            refs += "\n"

        refs += "See the [global index](../../index.md) for complete navigation.\n\n"

        return refs

    def generate_topic_navigation(self, folder_path: Path, keywords: dict) -> str:
        """Generate topic-based navigation from keywords."""
        topics = defaultdict(list)

        # Group keywords into topics
        ml_terms = ['regression', 'classification', 'clustering', 'neural', 'network']
        dl_terms = ['cnn', 'rnn', 'lstm', 'gan', 'autoencoder', 'convolution']
        data_terms = ['dataset', 'batch', 'preprocessing', 'augmentation']
        tf_terms = ['tensorflow', 'keras', 'estimator', 'layer']

        for kw in keywords:
            kw_lower = kw.lower()
            if any(term in kw_lower for term in ml_terms):
                topics['Machine Learning'].append(kw)
            if any(term in kw_lower for term in dl_terms):
                topics['Deep Learning'].append(kw)
            if any(term in kw_lower for term in data_terms):
                topics['Data Management'].append(kw)
            if any(term in kw_lower for term in tf_terms):
                topics['TensorFlow APIs'].append(kw)

        nav = ""
        for topic, kws in sorted(topics.items()):
            nav += f"**{topic}**: {', '.join(sorted(kws)[:10])}\n\n"

        if not nav:
            nav = "*Topic navigation available based on keywords in subtree.*\n\n"

        return nav

    def generate_all_documentation(self):
        """Main method to generate all documentation files."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPOSITORY DOCUMENTATION")
        print("="*80)

        # Scan repository first
        self.scan_repository()

        # Create base docs directory
        self.docs_dir.mkdir(exist_ok=True)

        # Generate per-file documentation
        print("\n[1/6] Generating per-file documentation...")
        file_count = 0
        for file_path in self.all_files:
            file_count += 1
            print(f"  Processing ({file_count}/{len(self.all_files)}): {file_path}")

            # Create directory structure in docs
            doc_file_dir = self.docs_dir / file_path.parent
            doc_file_dir.mkdir(parents=True, exist_ok=True)

            # Generate _docs.md
            docs_content = self.generate_file_docs(file_path)
            docs_path = doc_file_dir / f"{file_path.name}_docs.md"
            docs_path.write_text(docs_content, encoding='utf-8')

            # Generate _kw.md
            kw_content = self.generate_file_keywords(file_path)
            kw_path = doc_file_dir / f"{file_path.name}_kw.md"
            kw_path.write_text(kw_content, encoding='utf-8')

        print(f"  ✓ Generated documentation for {file_count} files")

        # Generate per-folder documentation
        print("\n[2/6] Generating per-folder indexes...")

        # Include root folder
        all_folders_including_root = [Path('.')] + self.all_folders

        folder_count = 0
        for folder_path in all_folders_including_root:
            folder_count += 1
            print(f"  Processing ({folder_count}/{len(all_folders_including_root)}): {folder_path}")

            doc_folder_dir = self.docs_dir / folder_path
            doc_folder_dir.mkdir(parents=True, exist_ok=True)

            # Generate index.md
            index_content = self.generate_folder_index(folder_path)
            (doc_folder_dir / "index.md").write_text(index_content, encoding='utf-8')

            # Generate doc.md
            doc_content = self.generate_folder_doc(folder_path)
            (doc_folder_dir / "doc.md").write_text(doc_content, encoding='utf-8')

            # Generate sub.md
            sub_content = self.generate_folder_subtree_keywords(folder_path)
            (doc_folder_dir / "sub.md").write_text(sub_content, encoding='utf-8')

        print(f"  ✓ Generated indexes for {folder_count} folders")

        # Generate global keywords.md
        print("\n[3/6] Generating global keyword index...")
        self.generate_global_keywords()
        print("  ✓ Generated keywords.md")

        # Generate comprehensive book
        print("\n[4/6] Generating comprehensive book...")
        self.generate_comprehensive_book()
        print("  ✓ Generated comprehensive_book.md")

        # Note: root index.md and doc.md are already generated in the folder loop

        print("\n" + "="*80)
        print("DOCUMENTATION GENERATION COMPLETE")
        print("="*80)
        print(f"\nStatistics:")
        print(f"  - Total files documented: {len(self.all_files)}")
        print(f"  - Total folders indexed: {len(all_folders_including_root)}")
        print(f"  - Documentation directory: {self.docs_dir}")
        print(f"  - Total keywords: {len(self.all_keywords)}")

    def generate_global_keywords(self):
        """Generate the global keywords.md file."""
        kw_doc = """# Global Keyword Index

## Usage

This is the master keyword index for the entire TensorFlow Examples repository. Use this index to find files related to specific concepts, APIs, or techniques.

Keywords are organized alphabetically for easy navigation. Each keyword links to all files where it appears.

## Alphabetical Index

"""

        # Group keywords by first letter
        by_letter = defaultdict(list)
        for kw in self.all_keywords:
            first_letter = kw[0].upper() if kw else '#'
            by_letter[first_letter].append(kw)

        # Generate A-Z sections
        for letter in sorted(by_letter.keys()):
            kw_doc += f"## {letter}\n\n"

            for kw in sorted(by_letter[letter], key=str.lower):
                kw_doc += f"### {kw}\n\n"
                kw_doc += "**Found in:**\n\n"

                for file_path in sorted(self.all_keywords[kw], key=str)[:20]:  # Limit to 20 files per keyword
                    docs_link = f"./{file_path.parent}/{file_path.name}_docs.md"
                    kw_link = f"./{file_path.parent}/{file_path.name}_kw.md"
                    kw_doc += f"- `{file_path}`\n"
                    kw_doc += f"  - [Documentation]({docs_link})\n"
                    kw_doc += f"  - [Keywords]({kw_link})\n"

                if len(self.all_keywords[kw]) > 20:
                    kw_doc += f"\n*... and {len(self.all_keywords[kw]) - 20} more files*\n"

                kw_doc += "\n"

        kw_doc += """
---

*This global keyword index was automatically generated for comprehensive repository navigation.*
"""

        (self.docs_dir / "keywords.md").write_text(kw_doc, encoding='utf-8')

    def generate_comprehensive_book(self):
        """Generate the comprehensive_book.md file."""
        book = """# TensorFlow Examples: The Comprehensive Book

## About This Book

This comprehensive book provides exhaustive documentation for the TensorFlow Examples repository. It serves as a complete reference guide, tutorial collection, and architectural overview of the entire codebase.

The repository contains educational examples demonstrating TensorFlow concepts from basic operations to advanced neural networks, covering both TensorFlow 1.x and 2.x.

---

# Part I: Project Overview

## Mission and Purpose

The TensorFlow Examples repository serves as an educational resource for developers learning TensorFlow. It provides:

- **Clear Examples**: Concise, well-commented code demonstrating key concepts
- **Progressive Learning**: Examples organized from beginner to advanced
- **Multiple Formats**: Both Jupyter notebooks and Python scripts
- **Version Coverage**: Support for both TensorFlow 1.x and 2.x

## Target Audience

- Machine learning beginners seeking practical TensorFlow examples
- Experienced developers transitioning to TensorFlow
- Students learning deep learning concepts
- Practitioners looking for implementation references

## Repository Structure

The repository is organized into several main categories:

1. **Introduction**: Basic TensorFlow operations and concepts
2. **Basic Models**: Fundamental machine learning algorithms
3. **Neural Networks**: Deep learning architectures and techniques
4. **Utilities**: Helper code for model management and visualization
5. **Data Management**: Data loading and preprocessing
6. **Multi-GPU**: Parallel and distributed training

---

# Part II: Global Architecture

## Repository Organization

```
TensorFlow-Examples/
├── examples/          # Python script examples (latest TF)
├── notebooks/         # Jupyter notebook examples (latest TF)
├── tensorflow_v1/     # TensorFlow 1.x specific code
├── tensorflow_v2/     # TensorFlow 2.x specific code
├── resources/         # Images and assets
└── docs/             # This comprehensive documentation
```

## Design Philosophy

The repository follows these design principles:

- **Simplicity**: Each example focuses on one concept
- **Clarity**: Code is heavily commented and explained
- **Completeness**: Examples include data loading, training, and evaluation
- **Modularity**: Examples can be run independently
- **Consistency**: Similar structure across all examples

## Technology Stack

- **Primary Framework**: TensorFlow (1.x and 2.x)
- **Languages**: Python 3
- **Notebook Environment**: Jupyter
- **Visualization**: TensorBoard, Matplotlib
- **Data**: MNIST, CIFAR-10, and other standard datasets

---

# Part III: Folder-by-Folder Deep Dive

"""

        # Add chapter for each major folder
        major_folders = [
            Path('.'),
            Path('examples'),
            Path('notebooks'),
            Path('tensorflow_v1'),
            Path('tensorflow_v2'),
            Path('resources')
        ]

        # Add numbered subdirectories
        numbered_dirs = [
            '0_Prerequisite',
            '1_Introduction',
            '2_BasicModels',
            '3_NeuralNetworks',
            '4_Utils',
            '5_DataManagement',
            '6_MultiGPU',
            '6_Hardware'
        ]

        for base in ['examples', 'notebooks', 'tensorflow_v1/examples', 'tensorflow_v1/notebooks',
                     'tensorflow_v2/notebooks']:
            for num_dir in numbered_dirs:
                folder_path = Path(base) / num_dir
                if folder_path in self.all_folders or any(str(f).startswith(str(folder_path)) for f in self.all_folders):
                    if folder_path not in major_folders:
                        major_folders.append(folder_path)

        for folder in sorted(set(major_folders), key=lambda x: str(x)):
            if folder in self.all_folders or folder == Path('.'):
                book += f"## Chapter: {folder if folder != Path('.') else 'Root Directory'}\n\n"
                book += self.generate_folder_role(folder)
                book += "\n"
                book += self.generate_folder_concepts(folder)
                book += "\n"

                # Link to folder documentation
                doc_link = f"./{folder}/doc.md" if folder != Path('.') else "./doc.md"
                index_link = f"./{folder}/index.md" if folder != Path('.') else "./index.md"

                book += f"For complete details, see:\n"
                book += f"- [Folder Documentation]({doc_link})\n"
                book += f"- [Folder Index]({index_link})\n\n"
                book += "---\n\n"

        book += """
# Part IV: File-by-File Reference

This section provides condensed references to all files in the repository. For full documentation, see individual `_docs.md` files.

"""

        # Group files by directory for organized presentation
        files_by_dir = defaultdict(list)
        for file_path in self.all_files:
            files_by_dir[file_path.parent].append(file_path)

        for directory in sorted(files_by_dir.keys(), key=str):
            book += f"## Files in: {directory if directory != Path('.') else 'Root'}\n\n"

            for file_path in sorted(files_by_dir[directory]):
                book += f"### {file_path.name}\n\n"
                book += f"- **Path**: `{file_path}`\n"
                book += f"- **Purpose**: {self.determine_file_purpose(file_path, '')}\n"
                book += f"- **Documentation**: [{file_path.name}_docs.md](./{file_path.parent}/{file_path.name}_docs.md)\n"
                book += f"- **Keywords**: [{file_path.name}_kw.md](./{file_path.parent}/{file_path.name}_kw.md)\n\n"

        book += """
---

# Part V: Patterns, Idioms, and Best Practices

## Common Patterns in TensorFlow

### 1. Data Pipeline Pattern

```python
# Load data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Preprocess and batch
dataset = dataset.shuffle(buffer_size).batch(batch_size)

# Use in training
for batch_x, batch_y in dataset:
    # Training step
    pass
```

### 2. Model Building Pattern

```python
# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### 3. Custom Training Loop Pattern

```python
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(num_epochs):
    for batch_x, batch_y in dataset:
        loss = train_step(batch_x, batch_y)
```

## Best Practices Demonstrated

1. **Use tf.data for Data Pipelines**: Efficient, scalable data loading
2. **Leverage Keras API**: High-level abstractions reduce boilerplate
3. **Enable Eager Execution**: Interactive development and debugging
4. **Use @tf.function**: Performance optimization with graph compilation
5. **Implement Proper Validation**: Monitor generalization during training
6. **Save Models Regularly**: Checkpoint frequently to prevent loss
7. **Visualize with TensorBoard**: Track metrics and debug issues
8. **Batch Processing**: Improve GPU utilization and training speed

---

# Part VI: Performance and Scaling

## Performance Considerations

### GPU Utilization

- Use batch processing to maximize GPU throughput
- Profile with TensorBoard to identify bottlenecks
- Consider mixed precision training for modern GPUs

### Multi-GPU Training

See examples in `6_MultiGPU/` for:
- Data parallelism
- Distributed training strategies
- Gradient aggregation across devices

### Memory Management

- Use `tf.data` with prefetching and caching
- Implement gradient accumulation for large batches
- Clear session state when running multiple experiments

## Scalability Patterns

1. **Data Parallelism**: Replicate model across GPUs
2. **Model Parallelism**: Split large models across devices
3. **Distributed Training**: Scale across multiple machines
4. **Mixed Precision**: Use FP16 for faster training

---

# Part VII: Security, Safety, and Reliability

## Security Considerations

### Data Privacy

- Be cautious with sensitive training data
- Models can memorize and leak training data
- Consider differential privacy for sensitive applications

### Model Security

- Validate inputs to prevent adversarial attacks
- Use secure model serving practices
- Monitor for unusual inference patterns

### Code Safety

- Validate file paths before loading data
- Sanitize user inputs in production
- Use environment variables for sensitive config

## Reliability Best Practices

1. **Version Control**: Track code and model versions
2. **Reproducibility**: Set random seeds, document dependencies
3. **Testing**: Validate model outputs and data pipelines
4. **Monitoring**: Track model performance in production
5. **Fallback**: Implement graceful degradation

---

# Part VIII: Extension and Maintenance

## How to Extend This Repository

### Adding New Examples

1. Choose appropriate category folder
2. Follow naming conventions: `descriptive_name.py`
3. Include comprehensive comments
4. Add both script and notebook versions
5. Update README with example description
6. Test thoroughly before committing

### Modifying Existing Examples

1. Maintain backward compatibility when possible
2. Update both TF1 and TF2 versions if applicable
3. Preserve educational clarity
4. Test across different TensorFlow versions
5. Update documentation

### Contributing Guidelines

- Follow PEP 8 style guidelines
- Include docstrings for functions and classes
- Add type hints where appropriate
- Ensure examples run on CPU (GPU optional)
- Keep dependencies minimal

## Maintenance Tasks

- Update for new TensorFlow versions
- Fix deprecated API usage
- Refresh datasets and links
- Improve performance and clarity
- Address reported issues

---

# Part IX: Glossary and Concept Index

## Key Concepts

### Activation Functions
- **ReLU**: Rectified Linear Unit, `f(x) = max(0, x)`
- **Sigmoid**: Logistic function, outputs [0, 1]
- **Softmax**: Normalized exponentials for multi-class

### Architectures
- **CNN**: Convolutional Neural Network
- **RNN**: Recurrent Neural Network
- **LSTM**: Long Short-Term Memory
- **GAN**: Generative Adversarial Network
- **Autoencoder**: Unsupervised encoding network

### Optimization
- **SGD**: Stochastic Gradient Descent
- **Adam**: Adaptive Moment Estimation
- **Learning Rate**: Step size for parameter updates
- **Batch Size**: Number of samples per gradient update

### Regularization
- **Dropout**: Random neuron deactivation
- **L1/L2**: Weight penalties
- **Batch Normalization**: Normalize layer inputs

### Evaluation
- **Accuracy**: Correct predictions / total predictions
- **Loss**: Measure of prediction error
- **Precision/Recall**: Classification metrics
- **Validation**: Held-out data for hyperparameter tuning

## TensorFlow APIs

### Core Operations
- `tf.constant()`: Create constant tensors
- `tf.Variable()`: Create trainable variables
- `tf.function()`: Convert functions to graphs
- `tf.GradientTape()`: Record operations for autodiff

### Keras API
- `tf.keras.Model`: Base model class
- `tf.keras.layers`: Pre-built layer types
- `tf.keras.optimizers`: Optimization algorithms
- `tf.keras.losses`: Loss functions

### Data API
- `tf.data.Dataset`: Input pipeline abstraction
- `.map()`: Apply transformations
- `.batch()`: Group into batches
- `.prefetch()`: Overlap data loading and training

---

# Conclusion

This comprehensive book provides a complete reference to the TensorFlow Examples repository. Whether you're just starting with TensorFlow or looking for advanced implementation patterns, these examples serve as a practical guide.

## Next Steps

1. **Beginners**: Start with `1_Introduction` examples
2. **Intermediate**: Explore `2_BasicModels` and `3_NeuralNetworks`
3. **Advanced**: Study multi-GPU and custom implementation examples
4. **Contributors**: Review contribution guidelines and add new examples

## Additional Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [TensorFlow API Reference](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)

---

*This comprehensive book was automatically generated to document every aspect of the TensorFlow Examples repository. For detailed file-specific documentation, refer to individual `_docs.md` files throughout the `/docs` directory.*

**Repository**: TensorFlow-Examples
**Documentation Generated**: Comprehensive documentation system with full coverage
**Total Files Documented**: """ + str(len(self.all_files)) + """
**Total Folders Indexed**: """ + str(len(self.all_folders) + 1) + """

---

© TensorFlow Examples Contributors - Educational Resource for Machine Learning Community
"""

        (self.docs_dir / "comprehensive_book.md").write_text(book, encoding='utf-8')

def main():
    """Main entry point for documentation generation."""
    print("TensorFlow Examples - Comprehensive Documentation Generator")
    print("=" * 80)

    # Get repository root (current directory)
    repo_root = os.getcwd()
    print(f"Repository root: {repo_root}")

    # Create generator
    generator = RepoDocumentationGenerator(repo_root)

    # Generate all documentation
    generator.generate_all_documentation()

    print("\n✓ Documentation generation complete!")
    print(f"\nView the documentation:")
    print(f"  - Comprehensive Book: docs/comprehensive_book.md")
    print(f"  - Global Index: docs/index.md")
    print(f"  - Keyword Index: docs/keywords.md")
    print()

if __name__ == "__main__":
    main()
