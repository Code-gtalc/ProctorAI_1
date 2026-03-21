# ProctorGuard AI Dashboard (React + Vite)

## Run Both Backend + Frontend In One Terminal

From repo root:

```bash
python run_all.py
```

This starts:

- Backend: `http://127.0.0.1:5000`
- Frontend: Vite dev URL (usually `http://127.0.0.1:5173`)

Press `Ctrl + C` once to stop both.

## Verification Flow

1. Login with **Name** and **Register Number** (Register Number is used as `user_id`)
2. Start Voice Enrollment
3. Complete voice module and it returns to home dashboard
4. Start Eye Calibration
5. Complete calibration module and it returns to home dashboard
6. Click Start Exam to enter main exam monitoring dashboard

## API Base URL

Default API target:

```bash
VITE_API_BASE_URL=http://127.0.0.1:5000
```

Optional override in `react_dashboard_app/.env`.
