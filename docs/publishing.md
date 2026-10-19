# Publishing to GitHub

Suggested repository slug: `Relative-Pose-Regresion-iMED-challenge` (preserves the requested spelling). The README title preserves the requested display name. This folder is deliberately separate from the original workspace and its empty `.git` directory.

After creating an empty GitHub repository under the intended account, add its URL and push:

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/Relative-Pose-Regresion-iMED-challenge.git
git push -u origin main
```

Do not initialize the remote with another README if pushing the prepared local history. Data, caches, third-party checkouts, checkpoints and environment secrets are ignored. Add external asset download instructions or release attachments separately if appropriate.
