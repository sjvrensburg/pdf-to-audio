"""Image processing functions for PDF to audio conversion."""

from .api import make_api_call
from .utils import clean_base64_image


def process_image(client, base64_image, model):
    """Describe an image using Mistral's vision capabilities."""
    # Clean the base64 string to avoid double prefixes
    clean_base64 = clean_base64_image(base64_image)
    
    # Validate that we have actual base64 content
    if not clean_base64:
        return "Image could not be processed - no valid base64 data"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Describe this mathematical diagram or figure in detail for text-to-speech conversion. Focus on what the image shows, any mathematical content, axes, curves, labels, or data presented. Be specific about mathematical elements like equations, graphs, charts, or diagrams. Keep the description clear and suitable for audio conversion."
                },
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{clean_base64}"
                }
            ]
        }
    ]
    
    try:
        response = make_api_call(client, model, messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Image could not be processed due to an error"


def replace_image_references(markdown, image_descriptions):
    """Replace image references using string replacement instead of regex."""
    # Create a mapping of image IDs to descriptions
    for image_id, description in image_descriptions.items():
        # Use simple string replacement to avoid regex issues
        # Look for the markdown image pattern
        patterns_to_try = [
            f"![{image_id}]({image_id})",
            f"![]({image_id})",
            f"![img]({image_id})",
            f"![image]({image_id})",
            f"![figure]({image_id})",
            f"![chart]({image_id})",
            f"![diagram]({image_id})"
        ]
        
        replacement = f"\n\nImage description: {description}\n\n"
        
        # Try each pattern for replacement
        for pattern in patterns_to_try:
            if pattern in markdown:
                markdown = markdown.replace(pattern, replacement)
                break
        else:
            # If no standard pattern found, try a more general approach
            # Look for any markdown image syntax containing the image_id
            lines = markdown.split('\n')
            for i, line in enumerate(lines):
                if f']({image_id})' in line and line.strip().startswith('!['):
                    lines[i] = replacement.strip()
                    break
            markdown = '\n'.join(lines)
    
    return markdown


def process_page(page, client, image_model):
    """Replace image references in markdown with their descriptions."""
    markdown = page['markdown']
    if 'images' not in page or not page['images']:
        return markdown
    
    # Process all images and create descriptions
    image_descriptions = {}
    
    for image in page['images']:
        image_id = image['id']
        try:
            description = process_image(client, image['image_base64'], image_model)
            image_descriptions[image_id] = description
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            image_descriptions[image_id] = f"[Image {image_id} could not be processed]"
    
    # Replace image references using string replacement
    try:
        markdown = replace_image_references(markdown, image_descriptions)
    except Exception as e:
        print(f"Error replacing image references: {e}")
        # Fallback: append all image descriptions at the end
        for image_id, description in image_descriptions.items():
            markdown += f"\n\nImage description for {image_id}: {description}\n\n"
    
    return markdown