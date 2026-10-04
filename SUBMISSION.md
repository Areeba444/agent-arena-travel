# Submission summary
Replace placeholders with your roll number. Copy these details into your submission document.

- Full name: Areeba Javed
- Roll number: 23I-0570
- Class / section: F
- University email: i230570@isb.nu.edu.pk
- GitHub username: Areeba444
- Agent name and domain: Travel Itinerary Builder (Travel Itinerary Planning)
- Private GitHub repository URL: TO FILL
- Final source commit hash: TO FILL
- Working public agent interface URL: https://agent-arena-travel.onrender.com/
- GET /health URL: https://agent-arena-travel.onrender.com/health
- POST /arena/run URL: https://agent-arena-travel.onrender.com/arena/run
- GET /arena/manifest URL: https://agent-arena-travel.onrender.com/arena/manifest
- API documentation URL: https://agent-arena-travel.onrender.com/docs
- Hosting provider: Render
- Default model and provider: openai/gpt-oss-20b via Groq
- Other available models: openai/gpt-oss-120b
- Example input and expected behavior:
  - Input: Task "Create an itinerary for my trip to Amsterdam" with external flight & hotel confirmation vouchers.
  - Expected behavior: Agent extracts booking records into sandbox state, executes `build_itinerary`, detects zero conflicts, and outputs status `completed` with the finalized schedule in `final_response`.
- Cold-start / restart limitations: Free-tier cloud instances sleep after inactivity (cold start ~30-50s). In-memory chat memory resets upon container restart.
- Instructor repository invitation status: TO FILL
- Public test result file: evaluation/results/public_test_results.json

Submit your zip archive and PDF document in Google Classroom, then click Turn in.
The ZIP must contain one folder including this file.
