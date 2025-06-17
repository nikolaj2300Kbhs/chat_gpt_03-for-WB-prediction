from flask import Flask, request, jsonify
import os
import re
import logging
import time
from openai import OpenAI
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Verify environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set")
    raise ValueError("OPENAI_API_KEY is not set")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define prompt template
template = """
Context: {context}

Historical Data (for reference, use only for general trends):
{historical_data}

Box Information: {box_info}

You are an expert in evaluating Goodiebox welcome boxes for their ability to attract new members in Denmark. Your task is to predict the daily intake (new members per day) for the box at a Customer Acquisition Cost (CAC) of 17.5 EUR. A regression model has predicted a daily intake of {predicted_intake} based on the Box Information. Use this as the primary basis and apply minimal adjustments based on the Box Information and general trends from historical data, then return the final predicted daily intake as a whole number.

**Step 1: Start with Predicted Intake**
- Use the regression model's predicted intake: {predicted_intake}.

**Step 2: Apply Minimal Adjustments**
- Adjust based on Box Information (e.g., retail value, premium products, ratings, free gift value).
- Use historical data only for general trends (e.g., high retail value increases intake).
- Adjustments are conservative, max total change ±5%:
  - +0.5% per 10 EUR retail value above 100 EUR (max +2%).
  - +0.5% per premium product above 3 (max +1.5%).
  - +0.25% per 10 EUR free gift value if rating >4.0 (max +1.5%).

**Step 3: Clamp the Final Value**
- Ensure intake is between 1 and 90 members/day.

**Step 4: Round to Whole Number**
- Round the clamped intake to the nearest whole number.

Return only the numerical value of the predicted daily intake (e.g., 10).
"""

def call_openai_api(prompt_text, model_name="o3-preview"):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1,
            max_completion_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

def predict_box_intake(context, box_info, predicted_intake, historical_data):
    try:
        logger.info(f"Processing prediction with context: {context[:100]}...")
        logger.info(f"Box info: {box_info[:100]}...")
        prompt_text = template.format(
            context=context,
            historical_data=historical_data,
            box_info=box_info,
            predicted_intake=predicted_intake
        )
        for attempt in range(3):
            try:
                result = call_openai_api(prompt_text)
                match = re.search(r'\d+', result)
                if match:
                    intake = round(float(match.group()))
                    intake = max(1, min(90, intake))
                    logger.info(f"Predicted intake: {intake}")
                    return intake
                else:
                    logger.warning(f"Invalid intake format: {result}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < 2:
                    time.sleep(0.5)
        raise ValueError("No valid intake value collected")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    try:
        data = request.get_json()
        if not all(k in data for k in ['box_info', 'context', 'predicted_intake']):
            logger.error("Missing required fields in request")
            return jsonify({'error': 'Missing box_info, context, or predicted_intake'}), 400
        intake = predict_box_intake(
            data['context'],
            data['box_info'],
            data['predicted_intake'],
            data.get('historical_data', '')
        )
        return jsonify({'predicted_intake': intake})
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'})

@app.route('/test_model', methods=['GET'])
def test_model():
    try:
        logger.info("Testing OpenAI model")
        result = call_openai_api("Test prompt to verify API access")
        logger.info(f"Test successful: {result}")
        return jsonify({'status': 'success', 'response': result})
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
