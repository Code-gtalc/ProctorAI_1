import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  LogOut,
  Mic,
  Radar,
  ShieldCheck,
  UserCircle2,
  Wifi,
} from "lucide-react";
import DashboardCard from "./components/DashboardCard";
import VerificationStepCard from "./components/VerificationStepCard";

const SESSION_KEY = "pgai_candidate_session_v1";
const examDetails = {
  name: "Data Structures Midterm",
  duration: "60 minutes",
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000").replace(/\/$/, "");
const apiUrl = (path) => `${API_BASE}${path}`;

function StatusLine({ icon: Icon, label, ok }) {
  return (
    <div className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2">
      <div className="flex items-center gap-2">
        <Icon size={16} className={ok ? "text-emerald-600" : "text-amber-500"} />
        <span className="text-sm text-slate-700">{label}</span>
      </div>
      <span
        className={`h-2.5 w-2.5 rounded-full ${
          ok
            ? "bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,0.15)]"
            : "bg-amber-500 shadow-[0_0_0_4px_rgba(245,158,11,0.15)]"
        }`}
      />
    </div>
  );
}

function ProgressTracker({ steps }) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="text-base font-semibold text-slate-900">Verification Progress</h3>
      <div className="flex flex-wrap items-center gap-2 md:gap-3">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium ${
                step.completed
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : step.current
                    ? "border-amber-200 bg-amber-50 text-amber-700"
                    : "border-slate-200 bg-slate-100 text-slate-500"
              }`}
            >
              {step.completed ? <CheckCircle2 size={14} /> : <span className="text-xs">...</span>}
              <span>{step.shortLabel}</span>
            </div>
            {index < steps.length - 1 ? <span className="text-slate-300">{">"}</span> : null}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function buildReturnUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set("resume", "1");
  return url.toString();
}

export default function ProctorGuardDashboard() {
  const [candidateName, setCandidateName] = useState("");
  const [registerNo, setRegisterNo] = useState("");
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const [voiceCompleted, setVoiceCompleted] = useState(false);
  const [faceVerified, setFaceVerified] = useState(false);
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState("Not Started");

  const [cameraReady, setCameraReady] = useState(false);
  const [micReady, setMicReady] = useState(false);
  const [internetReady, setInternetReady] = useState(false);

  const [loadingStatus, setLoadingStatus] = useState(false);
  const [startingExam, setStartingExam] = useState(false);
  const [uiMessage, setUiMessage] = useState("");
  const [uiError, setUiError] = useState("");

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed.registerNo) {
          setCandidateName(parsed.candidateName || "");
          setRegisterNo(parsed.registerNo || "");
          setIsLoggedIn(true);
        }
      }
    } catch (_) {
      // Ignore malformed local session data.
    }
  }, []);

  useEffect(() => {
    const t1 = setTimeout(() => setCameraReady(true), 500);
    const t2 = setTimeout(() => setMicReady(true), 900);
    const t3 = setTimeout(() => setInternetReady(true), 1200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  async function refreshBackendStatus() {
    if (!isLoggedIn || !registerNo.trim()) {
      return;
    }
    setLoadingStatus(true);
    setUiError("");
    try {
      const [enrollmentResp, gazeResp] = await Promise.all([
        fetch(apiUrl(`/api/enrollment/status/${encodeURIComponent(registerNo.trim())}`)),
        fetch(apiUrl(`/api/monitor/gaze?user_id=${encodeURIComponent(registerNo.trim())}`)),
      ]);

      const enrollment = await enrollmentResp.json().catch(() => ({}));
      const gaze = await gazeResp.json().catch(() => ({}));

      setVoiceCompleted(Boolean(enrollment.enrollment_complete));
      setCalibrationDone(Boolean(gaze.calibrated));
      setCalibrationStatus(String(gaze.status || "Not Started"));
    } catch (_) {
      setUiError("Unable to fetch backend status. Ensure Flask app is running on port 5000.");
    } finally {
      setLoadingStatus(false);
    }
  }

  useEffect(() => {
    refreshBackendStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, registerNo]);

  useEffect(() => {
    if (!isLoggedIn || !registerNo.trim()) return;
    const intervalId = window.setInterval(() => {
      refreshBackendStatus();
    }, 8000);
    return () => window.clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, registerNo]);

  const allDone = voiceCompleted && faceVerified && calibrationDone;

  const steps = useMemo(() => {
    const firstPending = !voiceCompleted ? "voice" : !faceVerified ? "face" : !calibrationDone ? "calibration" : "";
    return [
      { id: "voice", shortLabel: "Voice", completed: voiceCompleted, current: firstPending === "voice" },
      { id: "face", shortLabel: "Face", completed: faceVerified, current: firstPending === "face" },
      { id: "calibration", shortLabel: "Calibration", completed: calibrationDone, current: firstPending === "calibration" },
    ];
  }, [voiceCompleted, faceVerified, calibrationDone]);

  const progressPercent = Math.round((steps.filter((s) => s.completed).length / steps.length) * 100);

  function handleLogin(event) {
    event.preventDefault();
    const trimmedName = candidateName.trim();
    const trimmedReg = registerNo.trim();
    if (!trimmedName) {
      setUiError("Name is required.");
      return;
    }
    if (!trimmedReg) {
      setUiError("Register Number is required.");
      return;
    }
    setUiError("");
    setUiMessage("");
    localStorage.setItem(SESSION_KEY, JSON.stringify({ candidateName: trimmedName, registerNo: trimmedReg }));
    setIsLoggedIn(true);
  }

  function handleLogout() {
    localStorage.removeItem(SESSION_KEY);
    setIsLoggedIn(false);
    setCandidateName("");
    setRegisterNo("");
    setVoiceCompleted(false);
    setFaceVerified(false);
    setCalibrationDone(false);
    setUiMessage("");
    setUiError("");
  }

  async function ensureMonitorStarted() {
    let lastError = null;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const response = await fetch(apiUrl("/api/monitor/start"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: registerNo.trim() }),
        });
        const payload = await response.json().catch(() => ({}));
        const message = String(payload.error || payload.message || "");
        if ((!response.ok || !payload.ok) && !message.toLowerCase().includes("already running")) {
          throw new Error(message || "Unable to start monitor.");
        }
        return;
      } catch (error) {
        lastError = error;
        await new Promise((resolve) => window.setTimeout(resolve, 250 * (attempt + 1)));
      }
    }
    throw lastError instanceof Error ? lastError : new Error("Unable to start monitor.");
  }

  function handleVoiceEnrollmentStart() {
    if (!registerNo.trim()) {
      setUiError("Please login with register number first.");
      return;
    }
    const returnTo = encodeURIComponent(buildReturnUrl());
    window.location.assign(
      apiUrl(`/voice-enrollment?user_id=${encodeURIComponent(registerNo.trim())}&return_to=${returnTo}`),
    );
  }

  function handleFaceVerification() {
    setUiError("");
    setUiMessage("Face verification API pending. Marked as complete temporarily.");
    setFaceVerified(true);
  }

  async function handleCalibrationStart() {
    if (!registerNo.trim()) {
      setUiError("Please login with register number first.");
      return;
    }
    setUiError("");
    setUiMessage("");
    try {
      await ensureMonitorStarted();
      const returnTo = encodeURIComponent(buildReturnUrl());
      window.location.assign(
        apiUrl(
          `/monitor?user_id=${encodeURIComponent(registerNo.trim())}&mode=calibration_only&return_to=${returnTo}`,
        ),
      );
    } catch (error) {
      setUiError(error instanceof Error ? error.message : "Unable to start calibration.");
    }
  }

  async function handleStartExam() {
    if (!allDone || startingExam) return;
    setUiError("");
    setUiMessage("");
    setStartingExam(true);
    try {
      await ensureMonitorStarted();
      window.location.assign(apiUrl(`/monitor?user_id=${encodeURIComponent(registerNo.trim())}&mode=exam`));
    } catch (error) {
      setUiError(error instanceof Error ? error.message : "Unable to start exam monitor.");
      setStartingExam(false);
    }
  }

  if (!isLoggedIn) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-b from-slate-100 to-blue-50 p-4">
        <div className="w-full max-w-lg rounded-3xl border border-slate-200 bg-white p-8 shadow-xl">
          <div className="mb-6 flex items-center gap-2">
            <ShieldCheck className="text-blue-600" size={22} />
            <h1 className="text-xl font-bold text-slate-900">ProctorGuard AI Login</h1>
          </div>
          <form className="space-y-4" onSubmit={handleLogin}>
            <div>
              <label htmlFor="candidate-name" className="text-sm font-medium text-slate-700">
                Name
              </label>
              <input
                id="candidate-name"
                value={candidateName}
                onChange={(e) => setCandidateName(e.target.value)}
                className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-blue-500 focus:ring-2"
                placeholder="Enter full name"
              />
            </div>
            <div>
              <label htmlFor="register-no" className="text-sm font-medium text-slate-700">
                Register Number (Primary ID)
              </label>
              <input
                id="register-no"
                value={registerNo}
                onChange={(e) => setRegisterNo(e.target.value)}
                className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none ring-blue-500 focus:ring-2"
                placeholder="Enter register number"
              />
            </div>
            {uiError ? <p className="text-sm text-red-700">{uiError}</p> : null}
            <button
              type="submit"
              className="w-full rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700"
            >
              Continue
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 to-blue-50 text-slate-900">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 md:px-6">
          <div className="flex items-center gap-2">
            <ShieldCheck className="text-blue-600" size={22} />
            <span className="text-lg font-bold">ProctorGuard AI</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 md:flex">
              <UserCircle2 size={16} className="text-slate-500" />
              <span className="text-sm text-slate-700">
                Welcome, {candidateName} ({registerNo})
              </span>
            </div>
            <button
              onClick={handleLogout}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
            >
              <LogOut size={14} />
              Logout
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-6 px-4 py-6 md:px-6 lg:grid-cols-3">
        <section className="space-y-6 lg:col-span-2">
          <DashboardCard>
            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Welcome to ProctorGuard AI</h1>
            <p className="mt-2 text-sm text-slate-600 md:text-base">
              You are about to begin a proctored examination. Please complete all verification steps before proceeding.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="rounded-xl bg-blue-50 p-4">
                <p className="text-xs uppercase tracking-wide text-blue-700">Exam Name</p>
                <p className="mt-1 text-base font-semibold text-blue-900">{examDetails.name}</p>
              </div>
              <div className="rounded-xl bg-indigo-50 p-4">
                <p className="text-xs uppercase tracking-wide text-indigo-700">Duration</p>
                <p className="mt-1 text-base font-semibold text-indigo-900">{examDetails.duration}</p>
              </div>
            </div>
          </DashboardCard>

          <ProgressTracker steps={steps} />

          <DashboardCard title="Verification Steps" subtitle={`${progressPercent}% complete`}>
            <div className="mb-4 h-2 w-full overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-500"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <VerificationStepCard
                title="Voice Enrollment"
                description="Runs the existing voice enrollment module, then returns here."
                completed={voiceCompleted}
                buttonLabel="Start Voice Enrollment"
                onAction={handleVoiceEnrollmentStart}
              />
              <VerificationStepCard
                title="Face Verification"
                description="Will be connected to your face API later."
                completed={faceVerified}
                buttonLabel="Verify Face"
                onAction={handleFaceVerification}
              />
              <VerificationStepCard
                title="Camera Calibration"
                description="Runs gaze calibration module, then returns here."
                completed={calibrationDone}
                buttonLabel="Start Calibration"
                onAction={handleCalibrationStart}
              />
            </div>
          </DashboardCard>

          {uiError ? (
            <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{uiError}</p>
          ) : null}
          {uiMessage ? (
            <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">{uiMessage}</p>
          ) : null}

          <button
            type="button"
            onClick={handleStartExam}
            disabled={!allDone || startingExam}
            className="w-full rounded-2xl bg-blue-600 px-6 py-4 text-base font-semibold text-white shadow-lg shadow-blue-500/25 transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
          >
            {startingExam ? "Starting Exam..." : "Start Exam"}
          </button>
        </section>

        <aside className="space-y-6">
          <DashboardCard title="Exam Instructions">
            <ul className="list-disc space-y-2 pl-5 text-sm text-slate-700">
              <li>Ensure your camera and microphone are enabled</li>
              <li>Sit in a well-lit environment</li>
              <li>Do not leave your seat during the exam</li>
              <li>AI monitoring will be active throughout the exam</li>
            </ul>
          </DashboardCard>

          <DashboardCard title="System Check Panel">
            <div className="space-y-2">
              <StatusLine icon={Camera} label="Camera status" ok={cameraReady} />
              <StatusLine icon={Mic} label="Microphone status" ok={micReady} />
              <StatusLine icon={Wifi} label="Internet connection" ok={internetReady} />
              <StatusLine icon={Eye} label="Face visibility" ok={faceVerified} />
            </div>
          </DashboardCard>

          <DashboardCard title="Verification Sync">
            <div className="text-sm text-slate-600">Calibration status: {calibrationStatus}</div>
            <button
              type="button"
              onClick={refreshBackendStatus}
              disabled={loadingStatus}
              className="mt-3 rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loadingStatus ? "Refreshing..." : "Refresh Status"}
            </button>
          </DashboardCard>

          <DashboardCard title="AI Monitoring Active">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="flex items-center gap-2 rounded-xl bg-slate-50 p-3 text-sm">
                <Eye size={16} className="text-blue-600" />
                <span>Eye tracking</span>
              </div>
              <div className="flex items-center gap-2 rounded-xl bg-slate-50 p-3 text-sm">
                <Camera size={16} className="text-blue-600" />
                <span>Face detection</span>
              </div>
              <div className="flex items-center gap-2 rounded-xl bg-slate-50 p-3 text-sm">
                <Mic size={16} className="text-blue-600" />
                <span>Voice verification</span>
              </div>
              <div className="flex items-center gap-2 rounded-xl bg-slate-50 p-3 text-sm">
                <Radar size={16} className="text-blue-600" />
                <span>Behavior analysis</span>
              </div>
            </div>
          </DashboardCard>

          <DashboardCard className="border-red-200 bg-red-50">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 text-red-600" size={20} />
              <p className="text-sm font-medium text-red-700">
                Any suspicious activity detected by the AI system may lead to termination of the exam.
              </p>
            </div>
          </DashboardCard>
        </aside>
      </main>
    </div>
  );
}
