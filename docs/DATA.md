# Datasets: accounts, download and storage (cloud only, free)

The datasets are **not** kept on the laptop. They are downloaded inside Google Colab and stored in
your **private** Google Drive. Everything here is free.

Licence rules (all three datasets are for non-commercial research only):
- never commit any dataset file to Git,
- never make the Drive folder public or share the files outside the group,
- cite the datasets in the report.

---

## NightCity: no account needed (decision D28)

NightCity (4,297 real night street images with pixel labels) is downloaded automatically by the Colab
notebook from its public Google Drive links, together with the corrected NightCity-fine labels. You do
nothing except press *Run all*. Terms: non-commercial teaching and research only; cite Tan et al. (2021)
and Wei et al. (2023).

## Step 1. Create the accounts (do this today, approval can take days)

Register with your **personal email address** (decision D20: a university mailbox closes after you
graduate, and the project must keep working). Always write **"GLA University, Mathura, India"** as
your organisation/affiliation so the dataset owners can see it is academic use.

The account email only matters for downloading once. After that the data lives in your personal
Google Drive, so losing access to any email later does not affect the project.

**If Cityscapes refuses a personal email** (it checks academic users by hand), register once with
your GLA email just to download. The data still ends up in your personal Drive.

Website buttons may be named slightly differently from what is written here.

### ACDC (needed for M1)
1. Open <https://acdc.vision.ee.ethz.ch>.
2. Click **Register** (or **Login → Register**). Fill in your name, personal email and
   affiliation ("GLA University, Mathura, India").
3. Confirm your email if a confirmation link is sent, then log in.
4. Open the **Download** page and read and accept the licence terms.
5. **Accept the licence once:** click any package on the Download page and accept the licence
   dialog. You do not need to download anything: the Colab notebook downloads ACDC by itself with
   your username and password (stored in Colab Secrets).

### Cityscapes (needed for E2, can come a little later)
1. Open <https://www.cityscapes-dataset.com/register/>.
2. Fill in the form: username, password, email, full name, organisation
   ("GLA University, Mathura"), and a short purpose such as
   *"B.Tech final-year research project on semantic segmentation in low-light conditions;
   non-commercial academic use."*
3. Accept the terms and submit. Your account is **checked by hand**, so wait for the approval
   email (often 1–3 days, sometimes more). Check your spam folder.
4. After approval, log in once at <https://www.cityscapes-dataset.com/login/> to make sure it
   works. Remember the **username and password**: the Colab notebook needs them (Step 3).

### Dark Zurich (optional stretch goal, can wait)
1. Open <https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/> (if the link has moved, search
   "Dark Zurich dataset").
2. Accept the terms. Only the **validation set** (`Dark_Zurich_val_anon.zip`, 50 labelled night
   images) is needed.

## Step 2. Prepare Google Drive (free 15 GB)

1. Open <https://drive.google.com> with the Google account you will use for Colab.
2. Check the free space at the bottom left. This project needs about **5 GB** free (the 15 GB is
   shared with Gmail and Google Photos).
3. Create a folder named exactly `lowlight-seg` in **My Drive**. Keep it private (do not use
   "Share" or "Anyone with the link").

The folder will end up looking like this:
```
My Drive/lowlight-seg/
├── data/      acdc_night.zip, cityscapes_val.zip (only the parts we use)
├── code/      the project code (see docs/PROGRESS.md, decision D16)
└── results/   experiment outputs, written during runs
```

## Step 3. Download the data inside Colab

First put the code on Drive: on the laptop run `.\scripts\pack_code.ps1` in the project folder and
upload the created `lowlight-code.zip` to `My Drive/lowlight-seg/code/`.

Then open <https://colab.research.google.com>, choose **File → Upload notebook**, and pick
`notebooks/colab_data_setup.ipynb` from the project folder. Follow the notes at its top. It:
1. Read your ACDC and Cityscapes usernames and passwords from **Colab Secrets** (the key icon in
   Colab's left sidebar: `ACDC_USERNAME`, `ACDC_PASSWORD`, `CITYSCAPES_USERNAME`,
   `CITYSCAPES_PASSWORD`). They are never typed into the notebook or saved in any file.
2. Download the Cityscapes zips straight from the official website into Colab's temporary disk
   (the 11 GB zip never touches your laptop or your Drive).
3. Keep only the 500 **val** images and labels, zip them, and save the small zip to
   `My Drive/lowlight-seg/data/`.
4. Download ACDC the same way (through ACDC's own website API) and keep only the **night** train
   and val images and labels.
5. Run `check-data` to confirm the counts (ACDC night train 400, val 106; Cityscapes val 500).

Each later Colab session copies these zips from Drive and unzips them onto Colab's fast local
disk, because reading thousands of small files directly from Drive is very slow.

### Fallback: if an automatic download does not work
Some sites only allow downloads in a normal browser after login. In that case:
1. Download the zip in your browser.
2. Upload it to `My Drive/lowlight-seg/data/raw/` (drag and drop at drive.google.com).
3. Delete it from your laptop's Downloads folder.
4. The notebook extracts the parts we need and then deletes the large raw zip from Drive.

## Expected layout after unzipping on Colab

The loaders expect this layout (checked by `check-data`):
```
data/acdc/rgb_anon/night/{train,val}/<sequence>/*_rgb_anon.png
data/acdc/gt/night/{train,val}/<sequence>/*_gt_labelTrainIds.png
data/cityscapes/leftImg8bit/val/{frankfurt,lindau,munster}/*_leftImg8bit.png
data/cityscapes/gtFine/val/{frankfurt,lindau,munster}/*_gtFine_labelIds.png
data/darkzurich/rgb_anon/val/night/<sequence>/*_rgb_anon.png            (optional)
data/darkzurich/gt/val/night/<sequence>/*_gt_labelTrainIds.png          (optional)
```
If the real files are laid out differently, the loaders are adapted and the change is written
in `docs/PROGRESS.md`. Do not rename files by hand.

**Usage rule:** ACDC `night/train` is only for tuning (E3) and training the selector (E6).
ACDC `night/val` is only for the final reported numbers. ACDC `night/test` is not used.

## Storage summary

| Item | Where | Approx. size |
|---|---|---|
| Cityscapes full zip (11 GB) | Colab temporary disk only, deleted automatically | 0 on Drive |
| Cityscapes val images + labels | Drive | about 1–1.5 GB |
| ACDC night train + val | Drive | a few GB (exact size checked in M1) |
| Model weights | downloaded fresh in each Colab session from Hugging Face | 0 on Drive |
| Results | Drive, plus small CSVs copied into `results/` | under 1 GB |
