"""
API with JSON Validation using Pydantic - Complete Solution
Validate JSON input using Pydantic before processing
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
import json
import os

# ============================================================
# STEP 1 - PYDANTIC BASICS
# ============================================================

print("="*50)
print("PYDANTIC BASICS")
print("="*50)

class SimpleProduct(BaseModel):
    """A simple product model for validation."""
    name: str
    price: float
    quantity: int = 1

    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v

print("\n1. Valid data:")
try:
    product1 = SimpleProduct(name="Widget", price=10.99, quantity=5)
    print(f"  ✓ Valid: {product1.name} - ${product1.price}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n2. Invalid data (negative price):")
try:
    product2 = SimpleProduct(name="Widget", price=-10.99)
except Exception as e:
    print(f"  ✗ Validation error (expected): {e}")

print("\n✓ Pydantic basics working!")


# ============================================================
# STEP 2 - PRODUCT DATA MODELS
# ============================================================

print("\n" + "="*50)
print("PRODUCT DATA MODELS")
print("="*50)

class ProductRequest(BaseModel):
    """Model for incoming product listing requests."""
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    category: str = Field(..., min_length=1)
    description: Optional[str] = None
    additional_info: Optional[str] = None

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace')
        return v.strip()

    @validator('category')
    def category_must_be_valid(cls, v):
        allowed = ['electronics', 'clothing', 'food', 'furniture', 'other']
        if v.lower() not in allowed:
            raise ValueError(f'Category must be one of: {allowed}')
        return v.lower()


class ProductListingOutput(BaseModel):
    """Model for validated ChatGPT output."""
    title: str
    description: str
    key_features: List[str]
    suggested_price: float

print("✓ Product models created!")


# ============================================================
# STEP 3 - VALIDATING JSON INPUT
# ============================================================

print("\n" + "="*50)
print("VALIDATING JSON INPUT")
print("="*50)

from pydantic import ValidationError

def validate_product_from_file(filepath: str):
    """Load and validate a product from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        product = ProductRequest(**data)
        print(f"  ✓ Valid product: {product.name} - ${product.price}")
        return product

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON format error: {e}")
        return None
    except ValidationError as e:
        print(f"  ✗ Validation errors:")
        for error in e.errors():
            print(f"     - Field '{error['loc'][0]}': {error['msg']}")
        return None

print("\nTesting valid file:")
validate_product_from_file('valid_product.json')

print("\nTesting invalid file:")
validate_product_from_file('invalid_product.json')


# ============================================================
# STEP 4 - INTEGRATING WITH CHATGPT API
# ============================================================

print("\n" + "="*50)
print("INTEGRATING WITH CHATGPT API")
print("="*50)

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_product_listing(product: ProductRequest):
    """Validate first, then call ChatGPT to generate a listing."""

    prompt = f"""
    Create a product listing for:
    - Name: {product.name}
    - Price: ${product.price}
    - Category: {product.category}
    - Description: {product.description or 'Not provided'}
    - Additional Info: {product.additional_info or 'Not provided'}

    Return ONLY a JSON object with these exact fields:
    - title (string)
    - description (string)
    - key_features (list of strings)
    - suggested_price (float)

    Return ONLY the JSON, no extra text, no markdown.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content
    output_data = json.loads(raw_output)
    validated_output = ProductListingOutput(**output_data)
    return validated_output

print("\nTesting ChatGPT integration with valid product...")
test_product = ProductRequest(
    name="Wireless Headphones",
    price=79.99,
    category="electronics",
    description="High quality sound with noise cancellation"
)

try:
    result = generate_product_listing(test_product)
    print(f"  ✓ Title: {result.title}")
    print(f"  ✓ Description: {result.description}")
    print(f"  ✓ Features: {result.key_features}")
    print(f"  ✓ Suggested price: ${result.suggested_price}")
except Exception as e:
    print(f"  ✗ Error: {e}")


# ============================================================
# STEP 5 - HANDLING MULTIPLE REQUESTS
# ============================================================

print("\n" + "="*50)
print("HANDLING MULTIPLE REQUESTS")
print("="*50)

def process_batch_requests(filepath: str):
    """Process a batch of product requests from a JSON file."""

    with open(filepath, 'r') as f:
        requests_list = json.load(f)

    successes = []
    failures = []

    for i, item in enumerate(requests_list):
        try:
            product = ProductRequest(**item)
            successes.append(product)
            print(f"  ✓ Request {i+1}: '{product.name}' passed validation")
        except ValidationError as e:
            failures.append({"index": i+1, "data": item, "errors": e.errors()})
            print(f"  ✗ Request {i+1}: Failed - {e.errors()[0]['msg']}")

    print(f"\n  --- Batch Summary ---")
    print(f"  ✓ Successful: {len(successes)}")
    print(f"  ✗ Failed: {len(failures)}")

    return successes, failures

process_batch_requests('batch_requests.json')


# ============================================================
# STEP 6 - CLIENT REQUEST HANDLER
# ============================================================

print("\n" + "="*50)
print("CLIENT REQUEST HANDLER")
print("="*50)

def handle_client_request(payload: dict) -> dict:
    """
    Main entry point for all client requests.
    Validates input, processes, validates output.
    Returns a consistent response shape.
    """

    # Step 1: Validate input
    try:
        product = ProductRequest(**payload)
    except ValidationError as e:
        return {
            "status": "error",
            "stage": "input_validation",
            "errors": e.errors()
        }

    # Step 2: Process with ChatGPT
    try:
        result = generate_product_listing(product)
    except Exception as e:
        return {
            "status": "error",
            "stage": "processing",
            "errors": str(e)
        }

    # Step 3: Return success
    return {
        "status": "success",
        "product_name": product.name,
        "listing": result.dict()
    }

print("\nTest 1 - Valid payload:")
valid_payload = {
    "name": "Gaming Mouse",
    "price": 49.99,
    "category": "electronics",
    "description": "Ergonomic design with RGB lighting"
}
response = handle_client_request(valid_payload)
print(f"  Status: {response['status']}")
if response['status'] == 'success':
    print(f"  Listing title: {response['listing']['title']}")

print("\nTest 2 - Invalid payload:")
invalid_payload = {"name": "", "price": -1, "category": "unknown"}
response = handle_client_request(invalid_payload)
print(f"  Status: {response['status']} - Stage: {response['stage']}")

print("\n" + "="*50)
print("✓ ALL STEPS COMPLETED!")
print("="*50)