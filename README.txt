FIGURE FORMATTER - SETUP
=========================

1. Extract this whole folder somewhere simple, e.g.:
   C:\Users\ECO LOGIC\Downloads\figure-formatter-project
   (avoid OneDrive-synced folders if possible)

2. Open Command Prompt IN this folder:
   - Open the folder in File Explorer
   - Click the address bar, type: cmd
   - Press Enter

3. Create a virtual environment (one-time):
   python -m venv venv

4. Activate it (do this every time you open a new terminal):
   venv\Scripts\activate
   (you should see "(venv)" appear at the start of the line)

5. Install everything (one-time, takes a few minutes):
   pip install -r requirements.txt

6. Run whichever tool you want:

   python figure_formatter_pptx_app.py
       -> single image, web UI, review/edit labels before export to .pptx

   python figure_formatter_chat_local.py
       -> single image, chat-command UI (detect/list/set/export), no API

   python figure_formatter_batch_pptx.py
       -> put images in the "input" folder, run this, get .pptx files
          for all of them in the "output" folder

   python figure_formatter_chat_app.py
       -> free-form natural language chat (needs ANTHROPIC_API_KEY set,
          costs a small amount per use)

   python figure_formatter_app.py
       -> not meant to be run directly; the other scripts import from it

Files:
- figure_formatter_app.py          core OCR/detection functions (required by all others)
- figure_formatter_pptx_app.py     main tool: single image -> editable pptx
- figure_formatter_chat_local.py   chat-command version, fully local
- figure_formatter_chat_app.py     free-form AI chat version, needs API key
- figure_formatter_batch_pptx.py   batch processes a whole folder of images
- requirements.txt                 all Python packages needed
- input/                           put figures here for batch mode
