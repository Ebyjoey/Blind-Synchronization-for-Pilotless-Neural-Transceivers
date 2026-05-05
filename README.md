# RehabTrack 🏃‍♂️⚡

> AI-Powered Physical Rehabilitation Tracking Platform with Gamification

A production-ready SaaS application built with Next.js 14, TypeScript, Prisma, and Framer Motion.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

```bash
cp .env.local.example .env.local
```

Edit `.env.local` and set your database URL:

```env
DATABASE_URL="postgresql://postgres:password@localhost:5432/rehabtrack"
NEXTAUTH_SECRET="your-random-secret-string-at-least-32-chars"
NEXTAUTH_URL="http://localhost:3000"
```

### 3. Set up the database

Make sure PostgreSQL is running, then:

```bash
# Push schema to database
npm run db:push

# Generate Prisma client
npm run db:generate

# Seed with demo data
npm run db:seed
```

### 4. Start the dev server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 🔑 Demo Credentials

| Email | Password |
|---|---|
| `demo@rehabtrack.com` | `demo123` |

---

## 📁 Project Structure

```
rehabtrack/
├── app/
│   ├── api/
│   │   ├── activity/route.ts       # POST log activity, GET stats
│   │   ├── leaderboard/route.ts    # GET top 20 players
│   │   ├── user/route.ts           # POST signup, GET profile
│   │   └── auth/[...nextauth]/     # NextAuth handler
│   ├── auth/
│   │   ├── login/page.tsx          # Login form
│   │   └── signup/page.tsx         # Registration form
│   ├── dashboard/page.tsx          # Main dashboard
│   ├── activity/page.tsx           # Activity tracker
│   ├── leaderboard/page.tsx        # Rankings
│   ├── profile/page.tsx            # User profile
│   ├── globals.css                 # Global styles
│   ├── layout.tsx                  # Root layout
│   └── providers.tsx               # Client providers
├── components/
│   ├── ui/skeleton.tsx             # Loading skeleton
│   ├── layout/
│   │   ├── DashboardLayout.tsx     # Shell with sidebar + header
│   │   ├── Sidebar.tsx             # Animated nav sidebar
│   │   ├── Header.tsx              # Top bar with XP display
│   │   ├── LandingHero.tsx         # Public landing page
│   │   └── NotificationStack.tsx   # Toast notifications
│   ├── dashboard/
│   │   ├── DashboardContent.tsx    # Main dashboard view
│   │   ├── StatsCard.tsx           # Metric card
│   │   ├── ProgressRing.tsx        # Circular SVG progress
│   │   ├── XPBar.tsx               # Level progress bar
│   │   ├── ActivityFeed.tsx        # Recent activities list
│   │   └── ProfileContent.tsx      # Profile page view
│   ├── activity/
│   │   ├── ActivityContent.tsx     # Tracker with AI readouts
│   │   └── AIFeedbackCard.tsx      # Post-session AI coach card
│   └── gamification/
│       └── LeaderboardContent.tsx  # Full leaderboard with podium
├── hooks/
│   └── useActivityTimer.ts         # Timer hook
├── lib/
│   ├── auth.ts                     # NextAuth config
│   ├── prisma.ts                   # Prisma singleton
│   └── utils.ts                    # XP math, formatting, AI feedback
├── store/index.ts                  # Zustand stores
├── types/
│   ├── index.ts                    # Shared TypeScript types
│   └── next-auth.d.ts              # Session type augmentation
└── prisma/
    ├── schema.prisma               # Database models
    └── seed.ts                     # Demo data seeder
```

---

## 🎮 Features

### Authentication
- Credentials-based login/signup via NextAuth v4
- JWT session strategy
- Protected routes with server-side redirect
- Password strength indicator on signup

### Dashboard
- Real-time XP & level display in header
- Daily goal progress ring (SVG animated)
- Level XP bar with shimmer effect
- Stats cards: Total XP, Streak, Activities, Level
- Recent activity feed

### Activity Tracking (Mock AI)
- Select from 4 activity types: Walking, Stretching, Strength, Balance
- Live session timer
- Simulated AI intensity readings (updates every second)
- Rotating AI classification messages
- XP preview during session
- Post-session AI coach feedback

### Gamification
- XP system: quadratic level curve (`level² × 100`)
- 8 badges with rarity tiers (Common → Legendary)
- Auto-badge award on XP threshold
- Day streak tracking
- Level-up notification

### Leaderboard
- Top 20 players ranked by XP
- Visual podium for top 3
- "You" badge highlighting current user
- Flame streak indicators
- Polling every 30 seconds

### Profile
- Avatar, name, email display
- Level progress bar
- 4 key stats cards
- Badge collection grid with rarity colors
- Full activity history (last 12)

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS + glassmorphism |
| Animations | Framer Motion |
| Auth | NextAuth v4 (credentials) |
| Database | PostgreSQL via Prisma ORM |
| State | Zustand |
| Data Fetching | TanStack React Query |
| UI Components | Custom + shadcn/ui patterns |
| Fonts | DM Sans + Syne |
| Avatars | DiceBear API |

---

## 🗄 Database Schema

| Model | Key Fields |
|---|---|
| `User` | id, name, email, password, image |
| `Activity` | userId, type, duration, xpEarned, intensity |
| `Score` | userId, totalXp, level, streak, lastActive |
| `Badge` | name, description, icon, xpRequired, rarity |
| `UserBadge` | userId, badgeId, earnedAt |

---

## 📦 NPM Scripts

```bash
npm run dev          # Start dev server
npm run build        # Production build
npm run start        # Start production server
npm run db:push      # Push schema to DB
npm run db:generate  # Generate Prisma client
npm run db:seed      # Seed demo data
npm run db:studio    # Open Prisma Studio
```

---

## 🎨 Design System

- **Theme**: Dark glassmorphism with emerald/cyan/violet accents
- **Background**: Deep navy (`#070a12`) with ambient gradient orbs
- **Glass cards**: `rgba(255,255,255,0.03)` + `backdrop-filter: blur`
- **Primary**: Emerald `#22c55e` → Cyan `#22d3ee` gradient
- **Typography**: Syne (display/headings) + DM Sans (body)
- **Animations**: Framer Motion spring animations, stagger reveals

---

## 🔧 XP Formula

```
XP = baseRate[activityType] × (duration in minutes) × intensity × 10

Base rates:
  WALKING:    1.0
  STRETCHING: 0.8
  STRENGTH:   1.5
  BALANCE:    1.2
  IDLE:       0.1

Level threshold: level² × 100 XP
  Level 1 → 2:   100 XP
  Level 2 → 3:   400 XP
  Level 3 → 4:   900 XP
  Level 10 → 11: 10,000 XP
```
