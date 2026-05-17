import os
import base64
import json
import time
from pathlib import Path
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO

try:
    from datasets import load_dataset
    import pandas as pd
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def setup_client():
    api_key = "***REMOVED-SECRET***"
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized successfully!")
    return client


def load_product_dataset():
    print("\n" + "="*50)
    print("STEP 2: Loading product dataset...")
    print("="*50)

    sample_products = [
        {
            "id": 1,
            "name": "Wireless Bluetooth Headphones",
            "price": 79.99,
            "category": "Electronics",
            "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400",
            "image_path": None
        },
        {
            "id": 2,
            "name": "Running Sneakers",
            "price": 129.99,
            "category": "Footwear",
            "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
            "image_path": None
        },
        {
            "id": 3,
            "name": "Leather Backpack",
            "price": 89.99,
            "category": "Bags",
            "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400",
            "image_path": None
        }
    ]

    images_dir = Path("product_images")
    images_dir.mkdir(exist_ok=True)

    for product in sample_products:
        try:
            print(f"  Downloading image for: {product['name']}...")
            response = requests.get(product["image_url"], timeout=10)
            if response.status_code == 200:
                img_path = images_dir / f"product_{product['id']}.jpg"
                with open(img_path, "wb") as f:
                    f.write(response.content)
                product["image_path"] = str(img_path)
                print(f"  ✓ Saved!")
        except Exception as e:
            print(f"  Could not download image: {e}")

    print(f"\n✓ Dataset ready with {len(sample_products)} products!")
    return sample_products


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded


def test_image_encoding(products):
    print("\n" + "="*50)
    print("STEP 3: Testing image encoding...")
    print("="*50)

    for product in products:
        if product.get("image_path") and Path(product["image_path"]).exists():
            encoded = encode_image_to_base64(product["image_path"])
            print(f"✓ Encoded '{product['name']}'")
            print(f"  Length: {len(encoded)} characters")
            print(f"  Preview: {encoded[:40]}...")
            return True

    print("No valid image files found to test encoding.")
    return False


def create_product_listing_prompt(product_name, price, category):
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}

Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 relevant keywords)

Format your response as JSON with the following structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2", "Feature 3"],
    "keywords": "keyword1, keyword2, keyword3"
}}

Be specific about what you see in the image. Mention colors, materials, design elements, and any distinctive features."""

    return prompt


def parse_json_response(raw_response):
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    try:
        cleaned = raw_response
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    print("  Warning: Could not parse JSON. Returning raw text.")
    return {
        "title": "Parsing Error",
        "description": raw_response,
        "features": [],
        "keywords": ""
    }


def generate_product_listing(client, product):
    print(f"\n  Processing: {product['name']}...")

    if not product.get("image_path") or not Path(product["image_path"]).exists():
        print(f"  Warning: No image found for {product['name']}, skipping...")
        return None

    try:
        encoded_image = encode_image_to_base64(product["image_path"])

        prompt = create_product_listing_prompt(
            product_name=product["name"],
            price=product["price"],
            category=product["category"]
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
        )

        raw_response = response.choices[0].message.content
        print(f"  ✓ Got response from ChatGPT ({len(raw_response)} characters)")

        listing = parse_json_response(raw_response)

        if listing:
            listing["product_id"] = product["id"]
            listing["product_name"] = product["name"]
            listing["price"] = product["price"]
            listing["category"] = product["category"]
            print(f"  ✓ Generated listing: '{listing.get('title', 'No title')}'")

        return listing

    except Exception as e:
        print(f"  Error generating listing for {product['name']}: {e}")
        return None


def process_all_products(client, products, max_products=3, delay_seconds=2):
    print("\n" + "="*50)
    print(f"STEP 6: Processing {min(max_products, len(products))} products...")
    print("="*50)

    all_listings = []
    failed_products = []

    products_to_process = products[:max_products]

    for i, product in enumerate(products_to_process):
        print(f"\n[{i+1}/{len(products_to_process)}] Processing product #{product['id']}")

        listing = generate_product_listing(client, product)

        if listing:
            all_listings.append(listing)
        else:
            failed_products.append(product["name"])

        if i < len(products_to_process) - 1:
            print(f"  Waiting {delay_seconds} seconds before next request...")
            time.sleep(delay_seconds)

    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"✓ Successfully generated: {len(all_listings)} listings")
    if failed_products:
        print(f"✗ Failed: {len(failed_products)} products: {failed_products}")

    return all_listings


def save_results(listings, output_file="generated_listings.json"):
    if not listings:
        print("No listings to save.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(listings)} listings to '{output_file}'")


def display_listing(listing):
    print("\n" + "="*60)
    print(f"PRODUCT: {listing.get('product_name', 'Unknown')}")
    print("="*60)
    print(f"Generated Title: {listing.get('title', 'N/A')}")
    print(f"\nDescription:\n{listing.get('description', 'N/A')}")
    print(f"\nKey Features:")
    for feature in listing.get("features", []):
        print(f"  • {feature}")
    print(f"\nSEO Keywords: {listing.get('keywords', 'N/A')}")
    print("="*60)


def main():
    print("="*60)
    print("  AUTOMATED PRODUCT LISTING GENERATOR")
    print("  LAB | API Calling to ChatGPT")
    print("="*60)

    client = setup_client()
    products = load_product_dataset()

    if not products:
        print("Error: No products loaded. Exiting.")
        return

    test_image_encoding(products)

    print("\n" + "="*50)
    print("STEP 4: Prompt template preview...")
    print("="*50)
    sample_prompt = create_product_listing_prompt(
        product_name=products[0]["name"],
        price=products[0]["price"],
        category=products[0]["category"]
    )
    print(sample_prompt[:400] + "...\n")

    listings = process_all_products(
        client=client,
        products=products,
        max_products=3,
        delay_seconds=2
    )

    if listings:
        print("\n\nGENERATED PRODUCT LISTINGS PREVIEW:")
        for listing in listings:
            display_listing(listing)

    save_results(listings)

    print("\n🎉 Lab complete! Check 'generated_listings.json' for results.")


if __name__ == "__main__":
    main()