# SwingIQ — AI Golf Swing Analyzer

SwingIQ is a computer vision powered golf swing analysis platform that uses live webcam video and pose tracking to evaluate swing mechanics in real time. The application analyzes body movement, swing tempo, posture, and follow-through to provide instant feedback and performance insights for golfers.

## Features

* Live webcam-based golf swing tracking
* Real-time pose and motion detection
* Swing scoring and performance feedback
* Shoulder, hip, and arm angle analysis
* Swing tempo and follow-through tracking
* Shot tendency detection (slice, hook, fade, draw)
* Real-time feedback UI and analytics dashboard
* Practice history and performance tracking

## Tech Stack

* Next.js
* React
* TypeScript
* Tailwind CSS
* MediaPipe Pose
* TensorFlow.js
* Supabase
* PostgreSQL
* Vercel

## How It Works

1. User records or streams a golf swing through their webcam
2. MediaPipe detects body landmarks and tracks movement
3. Swing mechanics are analyzed using mathematical motion calculations
4. The system generates a swing score and feedback report
5. Results and session data are stored for progress tracking

## Metrics Analyzed

* Swing tempo
* Shoulder rotation
* Hip rotation
* Head movement
* Wrist path
* Follow-through angle
* Stability and balance
* Predicted shot shape

## Future Improvements

* Club path tracking
* Ball trajectory prediction
* 3D swing visualization
* AI-generated coaching recommendations
* Side-by-side comparison with professional golfers
* Mobile support
* Multiplayer training challenges

## Installation

```bash
git clone https://github.com/yourusername/swingiq.git
cd swingiq
npm install
npm run dev
```

## Inspiration

Inspired by modern sports analytics platforms and golf training systems, SwingIQ combines computer vision, real-time feedback, and motion analysis to create an interactive AI-powered golf coaching experience.
