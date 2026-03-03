# Regime Dashboard — Setup Guide
## Total cost: ₹0/month | Time: ~30 minutes

---

## WHAT THIS DOES

Every weekday at 10:00 AM IST, GitHub automatically:
1. Runs your Python classifier
2. Downloads fresh Nifty/VIX/breadth data
3. Updates regime_current.json + regime_history.json
4. Vercel detects the change and re-deploys your dashboard

You do nothing after setup.

---

## STEP 1 — Install tools (one time only)

Install Node.js from https://nodejs.org (choose LTS version)
Install Git from https://git-scm.com

Then open a terminal and run:
```
npm install -g vercel
```

---

## STEP 2 — Create GitHub account + repo

1. Go to https://github.com → Sign up (free)
2. Click "New repository"
3. Name it: regime-dashboard
4. Set to PUBLIC (required for free GitHub Actions)
5. Click "Create repository"

---

## STEP 3 — Upload this project to GitHub

In terminal, navigate to this folder and run:
```
git init
git add .
git commit -m "Initial regime dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/regime-dashboard.git
git push -u origin main
```
Replace YOUR_USERNAME with your GitHub username.

---

## STEP 4 — Give GitHub Actions permission to push

1. On GitHub → Your repo → Settings → Actions → General
2. Scroll to "Workflow permissions"
3. Select "Read and write permissions"
4. Click Save

---

## STEP 5 — Run classifier manually first (get initial data)

In terminal, from the project root:
```
pip install yfinance pandas numpy
python data_pipeline/run_classifier.py
```

This creates:
- public/regime_current.json
- public/regime_history.json

Then commit and push these files:
```
git add public/
git commit -m "Add initial regime data"
git push
```

---

## STEP 6 — Deploy to Vercel

In terminal, from the project root:
```
npm install
vercel
```

When prompted:
- Set up and deploy? → Y
- Which scope? → your username
- Link to existing project? → N
- Project name? → regime-dashboard (or anything)
- Directory? → ./  (just press Enter)
- Override settings? → N

Vercel gives you a URL like:
https://regime-dashboard-xyz.vercel.app

Your dashboard is LIVE. ✓

---

## STEP 7 — Connect Vercel to GitHub (auto-deploy)

1. Go to https://vercel.com → your project
2. Settings → Git → Connect Git Repository
3. Select your GitHub repo
4. Done — every time GitHub Actions pushes new JSON, Vercel auto-redeploys

---

## THAT'S IT.

Your dashboard now:
- Updates automatically every weekday at 10 AM IST
- Costs ₹0/month
- Requires zero maintenance

---

## ADDING REAL FII/DII DATA (optional upgrade)

1. Download from: https://www.nseindia.com/reports/fii-dii
2. Format the CSV with columns: date, FII_Net, DII_Net
3. Save as: data_pipeline/fii_dii_data.csv
4. The run_classifier.py script will automatically detect and use it

---

## TROUBLESHOOTING

**GitHub Actions failing?**
→ Check Settings → Actions → General → Workflow permissions = Read+Write

**Dashboard shows "Could not load regime data"?**
→ Run Step 5 again and push the JSON files

**Vercel build failing?**
→ Make sure you ran `npm install` before `vercel`

**Data looks stale?**
→ Go to GitHub → Actions tab → click your workflow → "Run workflow" manually
