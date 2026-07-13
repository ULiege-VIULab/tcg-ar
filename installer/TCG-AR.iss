; TCG-AR Windows installer (Inno Setup 6).
; Build with installer\build.ps1 (it supplies AppVersion/StageDir/VendorDir).
;
; Post-install provisioning (Python env, PyTorch, models, card DB) runs as a
; HIDDEN worker process supervised by an embedded progress page: the worker
; (bootstrap.ps1 / update_carddb.ps1, launched with -Gui) writes
; state\progress.json + per-step logs, and the wizard polls them to drive
; progress bars, a live log view, and inline error display with Retry.
; The worker is launched via explorer.exe: Inno enables the RedirectionGuard
; process mitigation (inherited by children), which makes uv's Python-version
; junctions fail with os error 448 ("untrusted mount point") - explorer puts
; the worker in a fresh, unmitigated process tree.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef StageDir
  #define StageDir "stage\app"
#endif
#ifndef VendorDir
  #define VendorDir "vendor"
#endif

[Setup]
AppId={{B7C3D1E5-4A29-4F8B-9C6E-2D7F5A1B8E30}
AppName=TCG-AR
AppVersion={#AppVersion}
AppVerName=TCG-AR {#AppVersion}
AppPublisher=The TCG-AR authors (University of Liege)
AppPublisherURL=https://github.com/ULiege-VIULab/tcg-ar
AppSupportURL=https://github.com/ULiege-VIULab/tcg-ar
DefaultDirName={userpf}\TCG-AR
DefaultGroupName=TCG-AR
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
OutputBaseFilename=TCG-AR-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\LICENSE
SetupIconFile=assets\tcg-ar.ico
UninstallDisplayIcon={app}\installer\tcg-ar.ico
UninstallDisplayName=TCG-AR
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Messages]
WelcomeLabel1=Welcome to the TCG-AR Setup Wizard
WelcomeLabel2=This will install TCG-AR %1 on your computer.%n%nTCG-AR - Real-Time Multi-View Augmented Reality for Trading Card Game Streaming - detects, orients and identifies trading cards from ordinary cameras and streams augmented feeds to OBS.%n%nAfter the files are copied, the AI components are downloaded (several GB) with the progress shown right here in this window. You need an internet connection and an NVIDIA GPU.

[Components]
Name: "models"; Description: "Pre-trained AI models (~590 MB download) - required"; Types: full compact custom; Flags: fixed
Name: "carddb"; Description: "Card database + sprites (several GB, built from the Pokemon TCG API - needs a free API key)"; Types: full
Name: "embeddings"; Description: "Pre-compute card embeddings (recommended: makes the first launch instant)"; Types: full

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#VendorDir}\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
#ifdef MmcvWheel
; Pre-built mmcv wheel for the Blackwell stack (no compiler on user machines).
Source: "{#MmcvWheel}"; DestDir: "{app}\tools"; Flags: ignoreversion
#endif
Source: "bootstrap.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "update_carddb.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "tailer.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "assets\tcg-ar.ico"; DestDir: "{app}\installer"; Flags: ignoreversion

[InstallDelete]
; On upgrade: scrub old code so renamed/deleted modules do not linger.
; Never touches app\assets, app\work_dirs, app\settings.yaml, app\mediamtx.
Type: filesandordirs; Name: "{app}\app\core"
Type: filesandordirs; Name: "{app}\app\inference"
Type: filesandordirs; Name: "{app}\app\scripts"
Type: filesandordirs; Name: "{app}\app\installation"
Type: filesandordirs; Name: "{app}\app\training"
Type: filesandordirs; Name: "{app}\app\evaluation"
Type: filesandordirs; Name: "{app}\app\tests"
Type: filesandordirs; Name: "{app}\app\docs"

[Icons]
; The "TCG-AR" launch shortcut is deliberately NOT created here: bootstrap.ps1
; creates it on success, so a half-finished setup never leaves a broken entry.
Name: "{group}\TCG-AR Setup (repair)"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\bootstrap.ps1"" -Repair"; IconFilename: "{app}\installer\tcg-ar.ico"; Comment: "Resume or repair the TCG-AR setup (downloads, Python environment)"

[UninstallDelete]
; Shortcuts created by bootstrap.ps1 (unknown to the uninstaller's file list).
Type: files; Name: "{group}\TCG-AR.lnk"
Type: files; Name: "{group}\TCG-AR - Update card database.lnk"
Type: dirifempty; Name: "{group}"
Type: files; Name: "{app}\installer\worker.lnk"

[Code]
const
  UNINST_KEY = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{B7C3D1E5-4A29-4F8B-9C6E-2D7F5A1B8E30}_is1';
  SHOWCMD_MIN_NOACTIVE = 7;   { SW_SHOWMINNOACTIVE }
  MODE_FRESH = 0;
  MODE_REPAIR = 1;
  MODE_UPDATE = 2;
  NOSTART_TIMEOUT_TICKS = 100; { ~60 s at 600 ms without progress.json }

var
  { existing-install detection / maintenance }
  MaintenancePage: TInputOptionWizardPage;
  HaveExisting, FullyProvisioned, HaveCardDb: Boolean;
  ExistingDir: String;
  { GPU page }
  GpuPage: TWizardPage;
  GpuInfoLabel, GpuWarnLabel, StackLabel: TNewStaticText;
  StackCombo: TNewComboBox;
  GpuDetected: Boolean;
  GpuName, GpuDriver: String;
  GpuCap: Integer;   { compute capability * 10, e.g. sm_120 -> 120 }
  { API key page }
  ApiKeyPage: TInputQueryWizardPage;
  { embedded progress page }
  ProgressPage: TWizardPage;
  PPTitle, PPStatus, PPError: TNewStaticText;
  PPOverall, PPSub: TNewProgressBar;
  PPLog: TNewMemo;
  PPRetry: TNewButton;
  TimerID: LongWord;
  WorkerActive, WorkerFinished: Boolean;
  WorkerPid: Integer;
  NoProgressTicks: Integer;
  LastTail: String;

function SetTimer(hWnd, nIDEvent, uElapse, lpTimerFunc: LongWord): LongWord;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(hWnd, nIDEvent: LongWord): LongWord;
  external 'KillTimer@user32.dll stdcall';

{ ----------------------------------------------------------------------- }
{ Small helpers                                                            }
{ ----------------------------------------------------------------------- }
function NextField(var S: String): String;
var
  P: Integer;
begin
  P := Pos(',', S);
  if P = 0 then begin
    Result := Trim(S);
    S := '';
  end else begin
    Result := Trim(Copy(S, 1, P - 1));
    Delete(S, 1, P);
  end;
end;

{ '12.0' -> 120, '8.6' -> 86, garbage -> -1 }
function ParseCap(S: String): Integer;
var
  P, Major, Minor: Integer;
begin
  Result := -1;
  P := Pos('.', S);
  if P = 0 then begin
    Major := StrToIntDef(Trim(S), -1);
    Minor := 0;
  end else begin
    Major := StrToIntDef(Trim(Copy(S, 1, P - 1)), -1);
    Minor := StrToIntDef(Copy(Trim(Copy(S, P + 1, 1)), 1, 1), 0);
  end;
  if Major >= 0 then
    Result := Major * 10 + Minor;
end;

function DriverMajor(S: String): Integer;
var
  P: Integer;
begin
  P := Pos('.', S);
  if P > 0 then
    Result := StrToIntDef(Trim(Copy(S, 1, P - 1)), 0)
  else
    Result := StrToIntDef(Trim(S), 0);
end;

{ Minimal extractors for the JSON files we write ourselves. }
function ExtractJsonString(const Json, Key: String): String;
var
  Pat: String;
  P, Q: Integer;
begin
  Result := '';
  Pat := '"' + Key + '":';
  P := Pos(Pat, Json);
  if P = 0 then begin
    Pat := '"' + Key + '": ';
    P := Pos(Pat, Json);
    if P = 0 then exit;
  end;
  P := P + Length(Pat);
  while (P <= Length(Json)) and ((Json[P] = ' ') or (Json[P] = '"')) do begin
    if Json[P] = '"' then begin
      P := P + 1;
      break;
    end;
    P := P + 1;
  end;
  Q := P;
  while (Q <= Length(Json)) and (Json[Q] <> '"') and (Json[Q] <> ',') and (Json[Q] <> '}') do begin
    if (Json[Q] = '\') and (Q < Length(Json)) then
      Q := Q + 2   { skip escaped char (\" \\ ...) }
    else
      Q := Q + 1;
  end;
  Result := Copy(Json, P, Q - P);
  StringChangeEx(Result, '\"', '"', True);
  StringChangeEx(Result, '\\', '\', True);
end;

function ExtractJsonInt(const Json, Key: String; Default: Integer): Integer;
begin
  Result := StrToIntDef(Trim(ExtractJsonString(Json, Key)), Default);
end;

function JsonEscape(const S: String): String;
var
  I: Integer;
begin
  Result := '';
  for I := 1 to Length(S) do begin
    if (S[I] = '\') or (S[I] = '"') then
      Result := Result + '\';
    Result := Result + S[I];
  end;
end;

function InstallMode: Integer;
begin
  Result := MODE_FRESH;
  { Silent installs are always plain file installs (deterministic; the
    maintenance page is never shown, so its default must not leak in). }
  if WizardSilent then exit;
  if HaveExisting and (MaintenancePage <> nil) then
    Result := MaintenancePage.SelectedValueIndex;
end;

{ ----------------------------------------------------------------------- }
{ GPU detection                                                            }
{ ----------------------------------------------------------------------- }
procedure DetectGpu;
var
  TmpFile, Line, CapStr: String;
  Lines: TArrayOfString;
  ResultCode, I, Cap: Integer;
begin
  GpuDetected := False;
  GpuCap := -1;
  GpuName := '';
  GpuDriver := '';
  TmpFile := ExpandConstant('{tmp}\gpu.csv');
  if not Exec(ExpandConstant('{cmd}'),
      '/C nvidia-smi --query-gpu=compute_cap,name,driver_version --format=csv,noheader > "' + TmpFile + '" 2>&1',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    exit;
  if ResultCode <> 0 then
    exit;
  if not LoadStringsFromFile(TmpFile, Lines) then
    exit;
  { Multiple GPUs: keep the most capable one. }
  for I := 0 to GetArrayLength(Lines) - 1 do begin
    Line := Lines[I];
    CapStr := NextField(Line);
    Cap := ParseCap(CapStr);
    if Cap > GpuCap then begin
      GpuCap := Cap;
      GpuName := NextField(Line);
      GpuDriver := NextField(Line);
    end;
  end;
  GpuDetected := GpuCap > 0;
end;

procedure UpdateGpuUi;
var
  Warn: String;
  DrvMaj: Integer;
begin
  if GpuDetected then
    GpuInfoLabel.Caption := 'Detected: ' + GpuName + '  (compute capability '
      + IntToStr(GpuCap div 10) + '.' + IntToStr(GpuCap mod 10)
      + ', driver ' + GpuDriver + ')'
  else
    GpuInfoLabel.Caption := 'No NVIDIA GPU detected (nvidia-smi not found or reported no device).';

  Warn := '';
  if not GpuDetected then
    Warn := 'TCG-AR requires an NVIDIA GPU with CUDA. Without one, the application will not work.'
  else if GpuCap < 75 then
    Warn := 'This GPU (compute capability below 7.5) is older than the supported generations; TCG-AR may not work.'
  else begin
    DrvMaj := DriverMajor(GpuDriver);
    if (StackCombo.ItemIndex = 0) and (DrvMaj > 0) and (DrvMaj < 576) then
      Warn := 'Your NVIDIA driver (' + GpuDriver + ') is older than 576, required for the CUDA 13.2 stack. Please update it at nvidia.com/drivers.'
    else if (StackCombo.ItemIndex = 1) and (DrvMaj > 0) and (DrvMaj < 522) then
      Warn := 'Your NVIDIA driver (' + GpuDriver + ') is older than 522, required for the CUDA 11.8 stack. Please update it at nvidia.com/drivers.';
  end;
  GpuWarnLabel.Caption := Warn;
end;

procedure StackComboChange(Sender: TObject);
begin
  UpdateGpuUi;
end;

{ ----------------------------------------------------------------------- }
{ Embedded progress page: worker launch + polling                          }
{ ----------------------------------------------------------------------- }
function ProgressJsonPath: String;
begin
  Result := ExpandConstant('{app}\state\progress.json');
end;

procedure StartWorker;
var
  Script, Args, Lnk: String;
  ResultCode: Integer;
begin
  { reset UI }
  PPTitle.Caption := 'Preparing the setup...';
  PPStatus.Caption := '';
  PPError.Caption := '';
  PPError.Visible := False;
  PPRetry.Visible := False;
  PPOverall.Position := 0;
  PPSub.Style := npbstMarquee;
  PPLog.Lines.Clear;
  WorkerActive := True;
  WorkerFinished := False;
  WorkerPid := 0;
  NoProgressTicks := 0;
  LastTail := '';
  WizardForm.NextButton.Enabled := False;
  WizardForm.BackButton.Enabled := False;

  DeleteFile(ProgressJsonPath);
  DeleteFile(ExpandConstant('{app}\state\live.txt'));
  DeleteFile(ExpandConstant('{app}\state\tail.txt'));
  if InstallMode = MODE_UPDATE then
    Script := ExpandConstant('{app}\installer\update_carddb.ps1')
  else
    Script := ExpandConstant('{app}\installer\bootstrap.ps1');
  Args := '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + Script + '" -Gui';
  if InstallMode = MODE_REPAIR then
    Args := Args + ' -Repair';

  { Launch hidden AND outside Setup's mitigated process tree (explorer). }
  Lnk := ExpandConstant('{app}\installer\worker.lnk');
  CreateShellLink(Lnk, 'TCG-AR setup worker',
    ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    Args, ExpandConstant('{app}'), '', 0, SHOWCMD_MIN_NOACTIVE);
  Exec(ExpandConstant('{win}\explorer.exe'), '"' + Lnk + '"', '', SW_SHOW, ewNoWait, ResultCode);
end;

procedure StopWorkerProcess;
var
  ResultCode: Integer;
begin
  if WorkerPid > 0 then
    Exec(ExpandConstant('{sys}\taskkill.exe'),
      '/PID ' + IntToStr(WorkerPid) + ' /T /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure StopTimer;
begin
  if TimerID <> 0 then begin
    KillTimer(0, TimerID);
    TimerID := 0;
  end;
end;

procedure FinishOk;
begin
  WorkerActive := False;
  WorkerFinished := True;
  StopTimer;
  PPOverall.Position := PPOverall.Max;
  PPSub.Style := npbstNormal;
  PPSub.Position := PPSub.Max;
  if InstallMode = MODE_UPDATE then begin
    PPTitle.Caption := 'Card database is up to date!';
    PPStatus.Caption := 'The new cards are picked up automatically the next time TCG-AR starts. Click Next to finish.';
  end else begin
    PPTitle.Caption := 'TCG-AR is ready!';
    PPStatus.Caption := 'Launch TCG-AR from the Start Menu. Windows Firewall will ask once on first launch - click "Allow access". Click Next to finish.';
  end;
  WizardForm.NextButton.Enabled := True;
end;

procedure FinishFail(const Msg: String);
begin
  WorkerActive := False;
  StopTimer;
  PPSub.Style := npbstNormal;
  PPStatus.Caption := 'Setup stopped.';
  PPError.Caption := 'Error: ' + Msg + #13#10
    + 'This is usually a network hiccup. Click Retry to resume where it stopped'
    + ' (already-finished steps are kept). Logs: ' + ExpandConstant('{app}\logs');
  PPError.Visible := True;
  PPRetry.Visible := True;
  WizardForm.NextButton.Enabled := False;
end;

procedure OnProgressTick(H: LongWord; Msg: LongWord; IdEvent: LongWord; Time: LongWord);
var
  Json: AnsiString;
  JsonS, Status, Name, Tail: String;
  Step, Total, Pct, I: Integer;
  LiveLines, TailLines: TArrayOfString;
begin
  if not WorkerActive then exit;

  if not LoadStringFromFile(ProgressJsonPath, Json) then begin
    NoProgressTicks := NoProgressTicks + 1;
    if NoProgressTicks > NOSTART_TIMEOUT_TICKS then
      FinishFail('The setup worker did not start. Check that PowerShell is allowed on this system, then Retry.');
    exit;
  end;
  NoProgressTicks := 0;
  JsonS := String(Json);

  Status := ExtractJsonString(JsonS, 'status');
  Name := ExtractJsonString(JsonS, 'name');
  Step := ExtractJsonInt(JsonS, 'step', 0);
  Total := ExtractJsonInt(JsonS, 'total', 1);
  WorkerPid := ExtractJsonInt(JsonS, 'pid', 0);
  if Total < 1 then Total := 1;

  if Status = 'failed' then begin
    FinishFail(ExtractJsonString(JsonS, 'message'));
    exit;
  end;
  if Status = 'done' then begin
    FinishOk;
    exit;
  end;

  { running }
  if Step > 0 then
    PPTitle.Caption := 'Step ' + IntToStr(Step) + ' of ' + IntToStr(Total) + ' - ' + Name
  else
    PPTitle.Caption := Name;
  if Step > 0 then
    PPOverall.Position := ((Step - 1) * PPOverall.Max) div Total
  else
    PPOverall.Position := 0;

  { The sidecar (tailer.ps1) distills the multi-MB step logs into two tiny
    UTF-8 files: live.txt = percent + the current progress line (the live
    tqdm bar), tail.txt = the last log lines. Just render them. }
  Pct := -1;
  if LoadStringsFromFile(ExpandConstant('{app}\state\live.txt'), LiveLines) then begin
    if GetArrayLength(LiveLines) >= 1 then
      Pct := StrToIntDef(Trim(LiveLines[0]), -1);
    if GetArrayLength(LiveLines) >= 2 then
      PPStatus.Caption := LiveLines[1];
  end;
  if Pct >= 0 then begin
    PPSub.Style := npbstNormal;
    PPSub.Position := Pct;
    { blend sub-progress into the overall bar }
    if Step > 0 then
      PPOverall.Position := ((Step - 1) * PPOverall.Max + (Pct * PPOverall.Max) div 100) div Total;
  end else
    PPSub.Style := npbstMarquee;

  if LoadStringsFromFile(ExpandConstant('{app}\state\tail.txt'), TailLines) then begin
    Tail := '';
    for I := 0 to GetArrayLength(TailLines) - 1 do
      Tail := Tail + TailLines[I] + #13#10;
    if Tail <> LastTail then begin
      LastTail := Tail;
      PPLog.Lines.Text := Tail;
    end;
  end;
end;

procedure RetryClick(Sender: TObject);
begin
  StartWorker;
  if TimerID = 0 then
    TimerID := SetTimer(0, 0, 600, CreateCallback(@OnProgressTick));
end;

{ ----------------------------------------------------------------------- }
{ Wizard construction                                                      }
{ ----------------------------------------------------------------------- }
procedure InitializeWizard;
var
  Y: Integer;
  PrevJson: AnsiString;
begin
  { --- detect an existing installation --- }
  HaveExisting := False;
  FullyProvisioned := False;
  HaveCardDb := False;
  if RegQueryStringValue(HKEY_CURRENT_USER, UNINST_KEY, 'InstallLocation', ExistingDir) then begin
    ExistingDir := RemoveBackslash(ExistingDir);
    HaveExisting := (ExistingDir <> '') and FileExists(ExistingDir + '\state\install.json');
    FullyProvisioned := HaveExisting and FileExists(ExistingDir + '\state\stack.json');
    HaveCardDb := HaveExisting and FileExists(ExistingDir + '\state\09-cards.done');
  end;

  { --- maintenance page (only shown when an install exists) --- }
  if HaveExisting then begin
    MaintenancePage := CreateInputOptionPage(wpWelcome, 'Existing installation found',
      'TCG-AR is already installed at ' + ExistingDir,
      'Choose what you would like to do:', True, False);
    MaintenancePage.Add('Install or upgrade TCG-AR (full setup)');
    MaintenancePage.Add('Repair / resume the setup (re-checks every component, resumable)');
    MaintenancePage.Add('Update the card database (new card sets - also refreshes sprites and embeddings)');
    if not HaveCardDb then
      MaintenancePage.CheckListBox.ItemEnabled[2] := False;
    if FullyProvisioned and HaveCardDb then
      MaintenancePage.Values[2] := True   { complete install: updating is the common case }
    else
      MaintenancePage.Values[1] := True;  { incomplete install: resume it }
  end;

  { --- GPU page (after the directory page) --- }
  GpuPage := CreateCustomPage(wpSelectDir, 'Graphics card',
    'TCG-AR runs its AI models on an NVIDIA GPU');

  GpuInfoLabel := TNewStaticText.Create(GpuPage);
  GpuInfoLabel.Parent := GpuPage.Surface;
  GpuInfoLabel.Top := 0;
  GpuInfoLabel.Width := GpuPage.SurfaceWidth;
  GpuInfoLabel.AutoSize := False;
  GpuInfoLabel.WordWrap := True;
  GpuInfoLabel.Height := ScaleY(28);

  StackLabel := TNewStaticText.Create(GpuPage);
  StackLabel.Parent := GpuPage.Surface;
  StackLabel.Top := GpuInfoLabel.Top + GpuInfoLabel.Height + ScaleY(12);
  StackLabel.Caption := 'Software stack to install (auto-selected from your GPU):';

  StackCombo := TNewComboBox.Create(GpuPage);
  StackCombo.Parent := GpuPage.Surface;
  StackCombo.Top := StackLabel.Top + StackLabel.Height + ScaleY(6);
  StackCombo.Width := GpuPage.SurfaceWidth;
  StackCombo.Style := csDropDownList;
  StackCombo.Items.Add('RTX 50 series / Blackwell  -  CUDA 13.2, Python 3.14');
  StackCombo.Items.Add('RTX 20 / 30 / 40 series (Turing, Ampere, Ada)  -  CUDA 11.8, Python 3.11');
  StackCombo.OnChange := @StackComboChange;

  GpuWarnLabel := TNewStaticText.Create(GpuPage);
  GpuWarnLabel.Parent := GpuPage.Surface;
  GpuWarnLabel.Top := StackCombo.Top + StackCombo.Height + ScaleY(16);
  GpuWarnLabel.Width := GpuPage.SurfaceWidth;
  GpuWarnLabel.AutoSize := False;
  GpuWarnLabel.WordWrap := True;
  GpuWarnLabel.Height := ScaleY(60);
  GpuWarnLabel.Font.Color := clRed;

  DetectGpu;
  if GpuDetected and (GpuCap >= 100) then
    StackCombo.ItemIndex := 0
  else
    StackCombo.ItemIndex := 1;
  UpdateGpuUi;

  { --- API key page (after the components page) --- }
  ApiKeyPage := CreateInputQueryPage(wpSelectComponents, 'Pokemon TCG API key',
    'A free API key is needed to build the card database',
    'Get a free key at  https://dev.pokemontcg.io  (sign up, then copy the key), and paste it below.'
    + #13#10#13#10
    + 'The card database downloads about 20,000 card images (several GB) and can take 1-3 hours. '
    + 'The progress is shown in this window at the end of the installation, and it resumes '
    + 'automatically if interrupted.');
  ApiKeyPage.Add('API key:', False);

  { --- embedded progress page (after file copy) --- }
  ProgressPage := CreateCustomPage(wpInstalling, 'Setting up TCG-AR',
    'Downloading and configuring the AI components');

  PPTitle := TNewStaticText.Create(ProgressPage);
  PPTitle.Parent := ProgressPage.Surface;
  PPTitle.Top := 0;
  PPTitle.Width := ProgressPage.SurfaceWidth;
  PPTitle.AutoSize := False;
  PPTitle.Height := ScaleY(16);
  PPTitle.Font.Style := [fsBold];
  PPTitle.Caption := 'Preparing...';

  PPOverall := TNewProgressBar.Create(ProgressPage);
  PPOverall.Parent := ProgressPage.Surface;
  PPOverall.Top := PPTitle.Top + PPTitle.Height + ScaleY(6);
  PPOverall.Width := ProgressPage.SurfaceWidth;
  PPOverall.Height := ScaleY(16);
  PPOverall.Min := 0;
  PPOverall.Max := 1000;

  PPStatus := TNewStaticText.Create(ProgressPage);
  PPStatus.Parent := ProgressPage.Surface;
  PPStatus.Top := PPOverall.Top + PPOverall.Height + ScaleY(8);
  PPStatus.Width := ProgressPage.SurfaceWidth;
  PPStatus.AutoSize := False;
  PPStatus.Height := ScaleY(14);
  PPStatus.Caption := '';

  PPSub := TNewProgressBar.Create(ProgressPage);
  PPSub.Parent := ProgressPage.Surface;
  PPSub.Top := PPStatus.Top + PPStatus.Height + ScaleY(4);
  PPSub.Width := ProgressPage.SurfaceWidth;
  PPSub.Height := ScaleY(12);
  PPSub.Min := 0;
  PPSub.Max := 100;

  PPLog := TNewMemo.Create(ProgressPage);
  PPLog.Parent := ProgressPage.Surface;
  PPLog.Top := PPSub.Top + PPSub.Height + ScaleY(8);
  PPLog.Width := ProgressPage.SurfaceWidth;
  PPLog.Height := ProgressPage.SurfaceHeight - PPLog.Top - ScaleY(58);
  PPLog.ReadOnly := True;
  PPLog.ScrollBars := ssVertical;
  PPLog.WordWrap := True;
  PPLog.Font.Name := 'Consolas';
  PPLog.Font.Size := 8;

  PPError := TNewStaticText.Create(ProgressPage);
  PPError.Parent := ProgressPage.Surface;
  PPError.Top := PPLog.Top + PPLog.Height + ScaleY(6);
  PPError.Width := ProgressPage.SurfaceWidth - ScaleX(90);
  PPError.AutoSize := False;
  PPError.WordWrap := True;
  PPError.Height := ScaleY(50);
  PPError.Font.Color := clRed;
  PPError.Visible := False;

  PPRetry := TNewButton.Create(ProgressPage);
  PPRetry.Parent := ProgressPage.Surface;
  PPRetry.Width := ScaleX(80);
  PPRetry.Height := ScaleY(26);
  PPRetry.Left := ProgressPage.SurfaceWidth - PPRetry.Width;
  PPRetry.Top := PPLog.Top + PPLog.Height + ScaleY(6);
  PPRetry.Caption := 'Retry';
  PPRetry.OnClick := @RetryClick;
  PPRetry.Visible := False;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if (MaintenancePage <> nil) and (PageID = MaintenancePage.ID) then begin
    Result := not HaveExisting;
    exit;
  end;
  { maintenance modes reuse the previous choices - skip the config pages }
  if InstallMode <> MODE_FRESH then begin
    if (PageID = wpLicense) or (PageID = wpSelectDir)
       or (PageID = GpuPage.ID) or (PageID = wpSelectComponents)
       or (PageID = ApiKeyPage.ID) then begin
      Result := True;
      exit;
    end;
  end;
  if PageID = ApiKeyPage.ID then
    Result := not WizardIsComponentSelected('carddb');
end;

procedure CurPageChanged(CurPageID: Integer);
var
  Json: AnsiString;
begin
  { Pre-fill the API key from a previous installation of the same dir. }
  if (CurPageID = ApiKeyPage.ID) and (ApiKeyPage.Values[0] = '') then begin
    if LoadStringFromFile(WizardDirValue + '\state\install.json', Json) then
      ApiKeyPage.Values[0] := ExtractJsonString(String(Json), 'api_key');
  end;
  { Silent installs are files-only (Inno still walks the pages internally,
    so this would otherwise fire even under /VERYSILENT). }
  if (CurPageID = ProgressPage.ID) and not WizardSilent then begin
    StartWorker;
    if TimerID = 0 then
      TimerID := SetTimer(0, 0, 600, CreateCallback(@OnProgressTick));
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Key: String;
  Http: Variant;
  Status: Integer;
begin
  Result := True;

  { Silent installs must not block on interactive validation. }
  if WizardSilent then
    exit;

  if CurPageID = GpuPage.ID then begin
    if not GpuDetected then
      Result := MsgBox('No NVIDIA GPU was detected on this computer.'#13#10#13#10
        + 'TCG-AR cannot run without an NVIDIA GPU with CUDA support. '
        + 'Are you sure you want to continue the installation anyway?',
        mbError, MB_YESNO or MB_DEFBUTTON2) = IDYES;
    exit;
  end;

  if CurPageID = ApiKeyPage.ID then begin
    Key := Trim(ApiKeyPage.Values[0]);
    if Key = '' then begin
      MsgBox('Please enter your Pokemon TCG API key (free at https://dev.pokemontcg.io), '
        + 'or go back and untick the "Card database" component.', mbError, MB_OK);
      Result := False;
      exit;
    end;
    { Best-effort live check; network problems never block the install. }
    try
      Http := CreateOleObject('WinHttp.WinHttpRequest.5.1');
      Http.Open('GET', 'https://api.pokemontcg.io/v2/types', False);
      Http.SetRequestHeader('X-Api-Key', Key);
      Http.SetTimeouts(5000, 5000, 5000, 5000);
      Http.Send('');
      Status := Http.Status;
      if (Status = 401) or (Status = 403) then
        Result := MsgBox('The API key was rejected by the Pokemon TCG API (HTTP '
          + IntToStr(Status) + '). Use it anyway?',
          mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
    except
      { offline or endpoint hiccup: accept the key, the worker will tell }
    end;
  end;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  if (CurPageID = ProgressPage.ID) and WorkerActive then begin
    Confirm := False;
    Cancel := MsgBox('Setup is still downloading and configuring TCG-AR.'#13#10#13#10
      + 'Abort now? You can resume later with "TCG-AR Setup (repair)" in the Start Menu '
      + '- finished steps are kept.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
    if Cancel then begin
      StopWorkerProcess;
      StopTimer;
      WorkerActive := False;
    end;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  StackName: String;
begin
  if InstallMode = MODE_UPDATE then begin
    Result := 'Update the card database of the TCG-AR installation at:' + NewLine
      + Space + ExistingDir + NewLine + NewLine
      + 'New card sets, sprites and the embedding cache will be refreshed.' + NewLine
      + 'Only missing data is downloaded; the progress is shown in this window.';
    exit;
  end;
  if InstallMode = MODE_REPAIR then begin
    Result := 'Repair / resume the TCG-AR installation at:' + NewLine
      + Space + ExistingDir + NewLine + NewLine
      + 'Every component is re-checked; finished parts are kept. '
      + 'The progress is shown in this window.';
    exit;
  end;
  if StackCombo.ItemIndex = 0 then
    StackName := 'CUDA 13.2 / Python 3.14 (Blackwell)'
  else
    StackName := 'CUDA 11.8 / Python 3.11 (Turing/Ampere/Ada)';
  Result := MemoDirInfo + NewLine + NewLine
    + 'GPU stack:' + NewLine + Space + StackName + NewLine + NewLine
    + MemoComponentsInfo + NewLine + NewLine
    + 'After the files are copied, the AI components are downloaded with the progress '
    + 'shown in this window (3-8 GB; the card database can take 1-3 hours).';
end;

{ ----------------------------------------------------------------------- }
{ Record the wizard choices for the worker scripts                         }
{ ----------------------------------------------------------------------- }
function BoolToJson(B: Boolean): String;
begin
  if B then Result := 'true' else Result := 'false';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  StateDir, Stack, Key, Json: String;
begin
  if CurStep <> ssPostInstall then exit;

  StateDir := ExpandConstant('{app}\state');
  ForceDirectories(StateDir);
  ForceDirectories(ExpandConstant('{app}\logs'));

  { Maintenance modes keep the existing install.json (same stack, key,
    components); only a fresh install/upgrade rewrites it. }
  if InstallMode <> MODE_FRESH then exit;

  if StackCombo.ItemIndex = 0 then
    Stack := 'blackwell'
  else
    Stack := 'cu118';

  Key := '';
  if WizardIsComponentSelected('carddb') then begin
    Key := Trim(ApiKeyPage.Values[0]);
    { Unattended installs can pass the key on the command line: /ApiKey=... }
    if Key = '' then
      Key := Trim(ExpandConstant('{param:ApiKey|}'));
  end;

  Json := '{'#13#10
    + '  "stack": "' + Stack + '",'#13#10
    + '  "gpu_name": "' + JsonEscape(GpuName) + '",'#13#10
    + '  "gpu_compute_cap": "' + IntToStr(GpuCap div 10) + '.' + IntToStr(GpuCap mod 10) + '",'#13#10
    + '  "gpu_driver": "' + JsonEscape(GpuDriver) + '",'#13#10
    + '  "components": {'#13#10
    + '    "models": ' + BoolToJson(WizardIsComponentSelected('models')) + ','#13#10
    + '    "carddb": ' + BoolToJson(WizardIsComponentSelected('carddb')) + ','#13#10
    + '    "embeddings": ' + BoolToJson(WizardIsComponentSelected('embeddings')) + #13#10
    + '  },'#13#10
    + '  "api_key": "' + JsonEscape(Key) + '"'#13#10
    + '}'#13#10;

  SaveStringToFile(StateDir + '\install.json', Json, False);
end;

procedure DeinitializeSetup;
begin
  StopTimer;
end;

{ ----------------------------------------------------------------------- }
{ Uninstall                                                                }
{ ----------------------------------------------------------------------- }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then begin
    { Always remove everything we provisioned (Python, venv, caches, logs). }
    DelTree(ExpandConstant('{app}\env'), True, True, True);
    DelTree(ExpandConstant('{app}\python'), True, True, True);
    DelTree(ExpandConstant('{app}\tools'), True, True, True);
    DelTree(ExpandConstant('{app}\state'), True, True, True);
    DelTree(ExpandConstant('{app}\logs'), True, True, True);
    RegDeleteValue(HKEY_CURRENT_USER, 'Environment', 'POKEMON_TCG_API_KEY');

    { Deleting the data requires an explicit interactive Yes; a silent
      uninstall always keeps it (it is several GB / hours to rebuild). }
    if (not UninstallSilent) and
       (MsgBox('Also delete the downloaded data (AI models, card database, embeddings)?'
        + #13#10#13#10'This is several GB that takes hours to rebuild. Choose "No" to keep it '
        + 'for a future reinstall.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
      DelTree(ExpandConstant('{app}\app'), True, True, True);
  end;
  if CurUninstallStep = usPostUninstall then
    RemoveDir(ExpandConstant('{app}'));   { removes the root only if empty }
end;
