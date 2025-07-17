Legal Claims & Client Churn Predictive Analytics System

This system helps predict client churn (loss of clients) and manage legal claims more intelligently.

It's made up of two parts:

    Client (Frontend): A web application built with React.js and TypeScript.

    Backend: A server built with Python (Flask) using a SQL database (Flask SQLAlchemy).

📋 Requirements Before You Begin

Please make sure you have the following software installed:

    Git
    (Used to clone the project repository.)
    ➔ Download from: https://git-scm.com/downloads
    ➔ During installation, make sure to select "Git from the command line and also from 3rd-party software" to ensure proper setup.

    Node.js v20.11.1
    (Used to run the Client/Frontend.)
    ➔ Download from: https://nodejs.org/

    Python v3.12.2
    (Used to run the Backend/Server.)
    ➔ Download from: https://www.python.org/downloads/
    ➔ During installation, make sure you select ✅ "Add Python to PATH" to ensure proper setup.

    Visual Studio Code (VS Code) (optional but recommended)
    (Makes it much easier to view and edit the project files.)

🗓️ Cloning the Repository

To clone this project, you'll need to open Git Bash in a folder where you want to save the project.
Opening Git Bash in a Folder Location

    Open File Explorer and navigate to the desired folder.

    Right-click on an empty space within that folder.

    Select "Git Bash Here" from the context menu. This will open a Git Bash terminal directly in that folder.

Once Git Bash is open in the correct location, run the following command to clone the project:

git clone https://github.com/micpana/Legal-Claims-And-Client-Churn-Predictive-Analytics-System.git

⚙️ Project Setup Options

NB: Project setup requires a working internet connection!

There are two setup methods available:

    Automated Setup (Recommended)

    Manual Setup (Optional)

🚀 Automated Setup (Using Provided Scripts)

For the automated setup, you'll be clicking on .bat files within your Windows File Explorer. These scripts will handle the necessary commands for you.

Before clicking on any scripts, first open the project folder using File Explorer and ensure you are inside the root of the project directory (where you see files like backend_install_dependencies.bat, frontend_install_on.bat, etc.).

1. Install Backend Dependencies

In File Explorer, click on backend_install_dependencies.bat.

2. Install Frontend Dependencies

In File Explorer, click on frontend_install_dependencies.bat.

3. Start Backend Server

In File Explorer, click on backend_on.bat.

4. Start Frontend Server

In File Explorer, click on frontend_on.bat.

5. AI Dataset Generation and Training (Required Once)

In File Explorer, click on generate_train_predict.bat.

This script will automatically generate datasets, train the AI models, and make predictions. You must click this script at least once.
🛠️ Manual Setup (Step-By-Step Instructions)

If you prefer to set up the project manually, you'll be using the VS Code terminal.
Opening the Terminal in VS Code

    Via Menu: Go to View > Terminal or Terminal > New Terminal.

    Keyboard Shortcut: Press Ctrl + ` (backtick).

Opening Additional Terminal Tabs in VS Code

To open additional terminal tabs (for example, one for the client, one for the API, and another for Git commands), click the + icon in the VS Code terminal pane.

To switch between terminal tabs, click on the tab names at the top of the terminal pane.
Client (Frontend Setup)

    Install Node.js 20.11.1:
    Download and install Node.js from https://nodejs.org/.

    If you don't have a terminal open already, open a new terminal in VS Code using the methods described above.

    In your VS Code terminal, navigate to the client folder:

    cd client

    In the same client terminal tab, install required packages (wait for it to finish installing):

    npm install

    In the same client terminal tab, start the Client app (wait until the server is up and running):

    npm run dev

    Access the Client:
    Open your browser and go to http://localhost:5173 or the URL displayed in your terminal.

Backend (Server Setup)

    Install Python 3.12.2:
    Download and install Python from https://www.python.org/downloads/.

    If you don't have a terminal open already, open a new terminal in VS Code using the methods described above.

    Open a new terminal tab in VS Code specifically for the API server (see "Opening Additional Terminal Tabs in VS Code" above).

    In this new API terminal tab, navigate to the api folder:

    cd api

    In the same API terminal tab, install required Python packages:

    pip install -r requirements.txt

    In the same API terminal tab, start the Backend server:

    python app.py

    Backend will start running locally:
    Typically accessible via http://127.0.0.1:5000. (no need to open this in your browser)

🔄 Updating the Project from Git

You can update your local project with the latest changes from the Git repository using one of two methods:
Automated Update

In File Explorer, navigate to the root of your project folder and click on get_updates.bat. This script will pull the latest changes from the remote repository.
Manual Update

    Open File Explorer and navigate to the root of your project folder.

    Right-click on an empty space within the project folder.

    Select "Git Bash Here" from the context menu.

    In the Git Bash terminal, run the following command:

    git pull

🗂️ Database

The backend uses a SQL database managed using Flask SQLAlchemy (SQLite by default).
👤 Default Users

The system comes with the following default users for testing purposes:

    Master Admin

        Email: admin@example.com

        Role: admin

        Password: AdminPass123

    Operations Manager

        Email: manager@example.com

        Role: manager

        Password: ManagerPass123

    Default User

        Email: user@example.com

        Role: user

        Password: UserPass123

📌 Notes

    Git is required to properly clone and update the project repository.

    VS Code is highly recommended for easy file navigation and editing, and for using the integrated terminal.

    Node.js is required for the Client (Frontend).

    Python 3 is required for the Backend (Server).

    Make sure both Client and Backend are running for the system to function correctly.

    Automated scripts are recommended for a faster and easier setup.

    The AI dataset generation and training script (generate_train_predict.bat) must be clicked at least once.

🛠️ Troubleshooting

    If you see errors about missing dependencies:

        Ensure you clicked backend_install_dependencies.bat and frontend_install_dependencies.bat OR

        Ran npm install inside the client folder.

        Ran pip install -r requirements.txt inside the api folder.

    If a port (like 5173 or 5000) is already in use:

        Stop other running applications using those ports, or configure the project to use a different port.

    If git is not recognized:

        Make sure Git was properly installed and added to your system’s PATH.