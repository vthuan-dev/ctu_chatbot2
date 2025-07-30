# CTU Chatbot on RunPod

This project provides a serverless deployment of a fine-tuned chatbot model for Can Tho University (CTU) using RunPod. The chatbot is designed to assist users with inquiries related to university admissions.

## Project Structure

- `runpod_handler.py`: Contains the serverless handler function for the CTU chatbot, including model loading and response generation.
- `Dockerfile`: Defines the Docker image for the project, specifying the base image and dependencies.
- `runpod_requirements.txt`: Lists the Python dependencies required for the project.
- `README.md`: Documentation for the project, including deployment instructions.

## Deployment Instructions

To deploy the fine-tuned model to RunPod serverless, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.

   ```bash
   git clone <repository-url>
   cd runpod-ctu-chatbot
   ```

2. **Build the Docker Image**: Use the provided Dockerfile to build the Docker image.

   ```bash
   docker build -t runpod-ctu-chatbot .
   ```

3. **Push the Docker Image to RunPod**: After building the image, push it to your RunPod account.

   ```bash
   runpod push runpod-ctu-chatbot
   ```

4. **Deploy the Model**: Use the RunPod interface to deploy the model. You can specify the necessary environment variables, such as `MODEL_NAME`, to point to your fine-tuned model.

5. **Test the Deployment**: Once deployed, you can test the chatbot by sending requests to the RunPod endpoint.

## Environment Variables

- `MODEL_NAME`: The name of the HuggingFace model to be used. Default is set to `thuanhero1/llama3-8b-finetuned-ctu`.

## Requirements

Ensure you have Docker installed on your machine and that you have access to RunPod for deployment.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.