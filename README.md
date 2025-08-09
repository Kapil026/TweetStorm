# Tweet Generator

A Flask web application that generates funny, viral-worthy tweets using AI and allows users to post them directly to their X (Twitter) account.

## Features

- 🤖 **AI-Powered Tweet Generation**: Uses Groq's LLM models to generate humorous tweets
- 🔄 **Iterative Optimization**: Automatically improves tweets based on AI feedback
- 🔐 **X OAuth Integration**: Secure login with X API for posting tweets
- 🎨 **Modern UI**: Clean, responsive interface with Twitter-like design
- 📱 **Mobile Friendly**: Works seamlessly on desktop and mobile devices

## Setup

### Prerequisites

- Python 3.10 or higher
- X Developer Account with API access
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Tweet-Generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or using uv
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory with:
   ```
   CLIENT_ID=your_x_client_id
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and go to `http://localhost:5000`

## Usage

1. **Generate Tweets**: Enter a topic and click "Generate & Optimize Tweet"
2. **Review Results**: See the generated tweet and optimization history
3. **Login to X**: Click "Login with X" to authenticate with your X account
4. **Post Tweets**: Once logged in, click "Post to My Twitter" to share your tweet

## API Endpoints

- `GET /` - Landing page
- `GET /tweet` - Tweet generator page
- `GET /login` - X OAuth login
- `GET /callback` - OAuth callback handler
- `GET /check-auth` - Check authentication status
- `GET /logout` - Logout user
- `POST /generate-tweet` - Generate AI tweet
- `POST /post-tweet` - Post tweet to X

## Architecture

- **Frontend**: HTML/CSS/JavaScript with modern UI design
- **Backend**: Flask web framework
- **AI**: Groq LLM models for tweet generation and optimization
- **Authentication**: X OAuth 2.0 for secure API access
- **Workflow**: LangGraph for iterative tweet optimization

## Security Notes

- Uses OAuth 2.0 for secure X API access
- Session-based authentication
- Environment variables for sensitive data
- CSRF protection through state parameter

## Development

The application consists of:
- `app.py` - Main Flask application
- `tweet.html` - Tweet generator interface
- `index.html` - Landing page
- `requirements.txt` - Python dependencies

## Troubleshooting

- **"AI models not available"**: Check your `GROQ_API_KEY` in `.env`
- **"Client ID not set"**: Ensure `CLIENT_ID` is set in `.env`
- **Login issues**: Verify your X API credentials and redirect URI

## License

This project is for educational purposes. Please respect X's API terms of service.
