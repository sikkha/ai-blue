# AI-blue

## Project Overview

AI-blue is a cutting-edge multihead AI integration platform designed to analyze and respond to complex scenarios. By leveraging a combination of AI models through advanced routing and weighting algorithms, AI-blue provides nuanced and intelligent insights into a wide range of dynamic real-world problems.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. AI-blue is built to be simple and straightforward, requiring only three main files to function:

1. `requirements.txt` - Lists all the necessary Python packages.
2. `ai_blue_prototype.py` - The main Python script for the AI-blue model.
3. `session_manager.py` - The user session management engine.

### Option

We've created the showcase for LLamaindex and Trulen to evaluate RAG and ChromaDB, but it doesn't use in the AI Blue yet. It will be explored to generate more document RAG Agent later.

4. `showcase_leaderboard.py` - Showcase for LLamaIndex, RAG, ChromaDB and Trulen evaluation leaderboard.

### Installation

1. Clone the repository:
   \```sh
   git clone https://github.com/yourusername/AI-blue.git
   \```
2. Navigate to the AI-blue directory:
   \```sh
   cd AI-blue
   \```
3. Install the required packages:
   \```sh
   pip install -r requirements.txt
   \```


### Running the Application

To run AI-blue, use the following command:

\```sh
nohup python session_manager.py &
\```

\```sh
nohup python ai_blue_prototype.py &
\```

Alternatively, you can use `screen` for session management.

AI-blue listens on port `192.0.0.1:7860` and the database session listens to port `192.0.0.1:5001`. You can experiment with the live model at:

- Local: `http://192.0.0.1:7860/`
- Public IP: `http://34.121.86.8:7860/`
- Short URL: [bit.ly/ai-blue](http://bit.ly/ai-blue)

## Concept and Algorithm

AI-blue utilizes a multihead AI approach where different AI models analyze the input data concurrently. Each model focuses on its area of expertise, and their outputs are then routed through a central "router" model. The router intelligently integrates these insights based on predefined weights and contextual understanding, producing a comprehensive and nuanced response.

### Key Features:

- **Dynamic Weighting:** Adjusts the influence of each sub-model in real-time based on the context.
- **Intelligent Routing:** Analyzes and directs tasks to the most suitable AI model.
- **Robust Integration:** Synthesizes various AI outputs into a cohesive response.

## Contributions

We welcome contributions and suggestions! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

## Acknowledgments

- Thanks to all the contributors who have invested their time into making AI-blue a valuable tool.
- Special thanks to the open-source community for providing the tools and libraries that make projects like this possible.
