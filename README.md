# Plant Stomata Counter-Analyser (PSCA)

> **🚀 Live App: Stomata Measurement Precision & QC Review**  
> Inspect automated stomata segmentation, verify biometric axes ($L_{\text{maj}}$, $L_{\text{min}}$, $P_w$, $G_w$), measure ground-truth with digital calipers, and evaluate precision across cytotypes on the CESNET cloud:  
> 👉 **[Open Stomata QC App (https://stomata-review-stenc.dyn.cloud.e-infra.cz)](https://stomata-review-stenc.dyn.cloud.e-infra.cz)** · **[Read the QC App Documentation](review_app.qmd)**

---

## ☁️ CESNET S3 Cloud Storage Connection Protocol

**Goal:** Connect our team's cloud storage (CESNET S3) to your local computer so it appears as a regular folder. This allows you to drag-and-drop files from tools like Fiji, Excel, or RStudio directly to the cloud without using the browser.

---

## 🛑 Step 0: Prerequisites (Install First)

Before you start, you need two things: **Rclone** (the connector software) and a **FUSE driver** (which lets Rclone "talk" to your file system).

### **Linux**
1.  **Install Rclone:**
    ```bash
    sudo apt install rclone
    ```
2.  **FUSE:** Usually pre-installed. If not:
    ```bash
    sudo apt install fuse3
    ```

### **Windows**
1.  **Install Rclone:** Download the [Rclone zip file](https://rclone.org/downloads/), unzip it, and place `rclone.exe` somewhere safe (e.g., `C:\rclone\`). Add this folder to your system PATH.
2.  **Install WinFsp:** Download and install [WinFsp](https://winfsp.dev/). **Crucial:** Rclone *cannot* mount a drive on Windows without this.

### **macOS**
1.  **Install Rclone:** Open Terminal and run:
    ```bash
    brew install rclone
    ```
    *(Requires Homebrew)*
2.  **Install macFUSE:** Download and install [macFUSE](https://osxfuse.github.io/).
    * *Note:* On newer Macs (M1/M2/M3), you may need to enable kernel extensions in Recovery Mode for macFUSE to work. Alternatively, verify if `fuse-t` is a suitable replacement for your version.

---

## ⚙️ Step 1: Configure the Connection (All Platforms)

This step is the same for everyone. Open your terminal (Command Prompt on Windows) and run:

```bash
rclone config
```

Follow the interactive menu using the Team Credentials (ask the Project Lead for keys):

    n (New remote)

    Name: cesnet_s3

    Storage: Select Amazon S3 Compliant Storage.

    Provider: Select Ceph.

    env_auth: false

    access_key_id: [Enter Team Access Key]

    secret_access_key: [Enter Team Secret Key]

    endpoint: https://s3.cl4.du.cesnet.cz

    acl: private

    Save: Press y.
    
    
## 🔗 Step 2: Connect the Drive

Choose the instructions for your operating system below.
🐧 Option A: Linux (Ubuntu/Debian)

The One-Time Command:
```bash
# 1. Create the mount folder
mkdir ~/cesnet_cloud

# 2. Mount it (runs in background)
rclone mount s3_cesnet: ~/cesnet_cloud --vfs-cache-mode writes --daemon
```
Make it Permanent (Auto-start on Login):

    Open "Startup Applications".

    Click Add.

    Command:
    ```bash
    rclone mount s3_cesnet: /home/YOUR_USER/cesnet_cloud --vfs-cache-mode writes --daemon
    ```
    
###  🪟 Option B: Windows

The One-Time Command (PowerShell):
```powershell
# 1. Create the folder (or pick a drive letter like Z:)
mkdir C:\cesnet_cloud

# 2. Mount it
rclone mount s3_cesnet: C:\cesnet_cloud --vfs-cache-mode writes
```
Note: This terminal window must stay open. If you close it, the drive disconnects.

Make it Permanent (Auto-start on Login):

    Create a new text file named mount_cesnet.bat.

    Paste this code inside:

    ```batch
    @echo off
start /b rclone mount s3_cesnet: Z: --vfs-cache-mode writes --no-console
    ```
    
(This mounts it as the Z: drive).

Press Win + R, type shell:startup, and press Enter.

Move your mount_cesnet.bat file into this folder. It will now run silently every time you log in.

### 🍎 Option C: macOS

The One-Time Command:
```bash
# 1. Create the mount folder
mkdir ~/cesnet_cloud

# 2. Mount it
rclone mount s3_cesnet: ~/cesnet_cloud --vfs-cache-mode writes --daemon
```
Make it Permanent (Auto-start on Login):

    Open "System Preferences" > "Users & Groups".

    Select your user and go to the "Login Items" tab.

    Click the "+" button and add a new script.

    Create a script file (e.g., mount_cesnet.sh) with the following content:

    ```bash
    #!/bin/bash
    rclone mount s3_cesnet: /Users/YOUR_USER/cesnet_cloud --vfs-cache-mode writes --daemon
    ```

    Make it executable:
    ```bash
    chmod +x /path/to/mount_cesnet.sh
    ```

    Add this script to your login items.
  
Save the app as "Connect CESNET".

Go to System Settings -> General -> Login Items and add "Connect CESNET" to the list.

### ⚠️ Important Best Practices

    Don't Edit in Place: Avoid opening a huge video file directly from the cloud drive to edit. Copy it to your desktop first, edit, then copy it back. (Small text/image edits are fine).

    Wait for the Upload: When you drag a file into the folder, it may take a few seconds to upload. Don't immediately shut down your computer.

    If it freezes: If the folder stops responding (e.g., internet loss), unmount it:

        Linux: fusermount -u ~/cesnet_cloud

        Windows: Ctrl+C in the terminal or restart.

        Mac: umount ~/cesnet_cloud
