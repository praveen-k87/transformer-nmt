import subprocess
import sys
from pathlib import Path

def convert_md_to_pdf():
    md_file = Path("report/Project_Report.md")
    pdf_file = Path("report/Project_Report.pdf")
    
    if not md_file.exists():
        print(f"Error: {md_file} not found.")
        return

    # Try using pandoc if installed
    try:
        print("Attempting to use pandoc...")
        subprocess.run(["pandoc", str(md_file), "-o", str(pdf_file), "--pdf-engine=xelatex", "-V", "geometry:margin=1in"], check=True)
        print("Successfully generated PDF using pandoc.")
        return
    except FileNotFoundError:
        print("pandoc not found on system path.")
    except subprocess.CalledProcessError as e:
        print(f"pandoc failed: {e}")

    # Fallback to npx md-to-pdf
    try:
        print("Attempting to use npx md-to-pdf...")
        subprocess.run(["npx", "--yes", "md-to-pdf", str(md_file)], check=True)
        print("Successfully generated PDF using npx md-to-pdf.")
        return
    except subprocess.CalledProcessError as e:
        print(f"npx md-to-pdf failed: {e}")

if __name__ == "__main__":
    convert_md_to_pdf()
