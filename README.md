# open_deep_research_by_autogen

Open Deep Research Solution by AutoGen

## Getting Started

Follow these steps to set up and use this repository.

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.10 or higher
- pip (Python package manager)

### Clone the Repository

To clone this repository, run the following command in your terminal:

```bash
git clone https://github.com/huqianghui/open_deep_research_by_autogen.git
cd open_deep_research_by_autogen
```

### Install Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Before running the application, you need to configure the environment variables. Follow these steps:

1. Locate the `.env_template` file in the root directory of the project.
2. Rename the file to `.env` by running the following command in your terminal:
   ```bash
   mv .env_template .env
   ```
3. Open the `.env` file and update the placeholder values with your actual credentials. For example:
   ```env
   AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
   AZURE_OPENAI_API_VERSION="2024-08-01-preview"
   AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
   AZURE_OPENAI_API_KEY="your-api-key"
   BING_SEARCH_KEY="your-bing-search-key"
   ```

The application will automatically load these environment variables when it starts.

### Run the Application

To start the application, execute the following command:

```bash
chainlit run app.py
```

### View Results

Once the application is running, follow the instructions provided in the terminal to interact with the system. Results will be displayed in the terminal or saved in the `public/` directory, depending on the functionality you use.

- **Custom CSS**: The `public/custom.css` file contains styles for the application.
- **Icons**: Icons are stored in the `public/icons/` directory.
- **PDFs**: Generated or downloaded PDFs are saved in the `public/pdfs/` directory.

### Project Structure

Here is an overview of the project structure:

```
app.py                # Main application entry point
config.py             # Configuration settings
requirements.txt      # Python dependencies
agents/               # Core logic and tools
  deep_research/      # Deep research agent
    main.py           # Main script for the agent
    tools/            # Utility tools for the agent
public/               # Public assets
  custom.css          # Custom styles
  icons/              # Icons used in the application
  pdfs/               # Generated or downloaded PDFs
```

### Contributing

If you would like to contribute to this project, feel free to submit a pull request or open an issue.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
