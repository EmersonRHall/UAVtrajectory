%% Week 9 — Full Randomization + Robust Video/Legend + UNSTICK (best solution)
%% Inputs: week6_dataset.mp4
%% Outputs: week9_demo.mp4, week9_final_path.png, week9_metrics.csv

clear; clc; close all; clear fig;
rng('shuffle');   % new layout every run

%% --- Load Week 6 dataset & ridge policy ---
S = load('week6_dataset.mat');
X  = S.features;  Y  = S.actions;
itr = S.split.train(:);  iva = S.split.val(:);
Xtr = X(itr,:);   Ytr = Y(itr,:);
Xva = X(iva,:);   Yva = Y(iva,:);

std_tr    = std(Xtr,0,1);
keep_cols = std_tr > 1e-8;
Xtr = Xtr(:,keep_cols);  Xva = Xva(:,keep_cols);
Xtr_b=[Xtr,ones(size(Xtr,1),1)];
Xva_b=[Xva,ones(size(Xva,1),1)];
XtX=Xtr_b'*Xtr_b; XtY=Xtr_b'*Ytr; I=eye(size(Xtr_b,2)); I(end,end)=0;
lambdas=logspace(-4,2,20); bestVa=inf; W_best=[];
for lam=lambdas
    W=(XtX+lam*I)\XtY;
    va_rmse=sqrt(mean(sum((Xva_b*W-Yva).^2,2)));
    if va_rmse<bestVa, bestVa=va_rmse; W_best=W; end
end

%% --- World & bounds ---
gridRes=0.20; 
safetyMargin = 0.70;
xyMin=[-10 -10]; xyMax=[10 10];
startXY=[-8,-8];

% Grid
xv=xyMin(1):gridRes:xyMax(1); yv=xyMin(2):gridRes:xyMax(2);
[XC,YC]=meshgrid(xv,yv); [mapRows,mapCols]=size(XC);
world2grid=@(xy)[ round((xy(2)-xyMin(2))/gridRes)+1, round((xy(1)-xyMin(1))/gridRes)+1 ];
grid2world=@(rc)[ xyMin(1)+(rc(:,2)-1)*gridRes, xyMin(2)+(rc(:,1)-1)*gridRes ];

%% --- Random layout generator ---
maxTries = 25;
goalXY   = [8,2]; 
haveLoS  = exist('isLineFree','file')==2;

for attempt = 1:maxTries
    % Randomize main vertical blue wall
    wall_thickness = max(0.30, min(0.60, 0.45 + 0.12*randn()));
    x_wall_center  = 0.0 + 1.2*randn();
    x_left  = x_wall_center - wall_thickness/2;
    x_right = x_wall_center + wall_thickness/2;

    % Randomize top lip height and wall polygon
    wall_top_y = 4.6 + 1.8*rand();        % [4.6, 6.4]
    blueWall = [x_left xyMin(2); x_right xyMin(2); x_right wall_top_y; x_left wall_top_y];

    % Randomize gray box
    gw = 2.2 + 1.8*rand(); gh = 2.0 + 1.6*rand();
    gxc = -1.0 + 7.0*rand();  gyc = 0.8 + 6.2*rand();
    gx1 = max(xyMin(1)+0.8, gxc - gw/2); gx2 = min(xyMax(1)-0.8, gxc + gw/2);
    gy1 = max(-0.8,            gyc - gh/2); gy2 = min(xyMax(2)-0.8, gyc + gh/2);
    grayBox  = [gx1 gy1; gx2 gy1; gx2 gy2; gx1 gy2];

    % Extra thin walls (3–6)
    nExtra = randi([3 6]);
    extraPolys = cell(1,nExtra);
    for i=1:nExtra
        if rand()<0.50
            % Thin wall (axis-aligned)
            if rand()<0.5 % vertical
                cx = (xyMin(1)+1) + (xyMax(1)-2 - (xyMin(1)+1))*rand();
                h  = 2.0 + 6.0*rand();
                w  = 0.35 + 0.25*rand();
                cy = (xyMin(2)+1) + (xyMax(2)-2 - (xyMin(2)+1))*rand();
                x1 = max(xyMin(1)+0.5, cx - w/2); x2 = min(xyMax(1)-0.5, cx + w/2);
                y1 = max(xyMin(2)+0.5, cy - h/2); y2 = min(xyMax(2)-0.5, cy + h/2);
            else        % horizontal
                cy = (xyMin(2)+1) + (xyMax(2)-2 - (xyMin(2)+1))*rand();
                w  = 2.0 + 6.0*rand();
                h  = 0.35 + 0.25*rand();
                cx = (xyMin(1)+1) + (xyMax(1)-2 - (xyMin(1)+1))*rand();
                x1 = max(xyMin(1)+0.5, cx - w/2); x2 = min(xyMax(1)-0.5, cx + w/2);
                y1 = max(xyMin(2)+0.5, cy - h/2); y2 = min(xyMax(2)-0.5, cy + h/2);
            end
        else
            % Box
            cx = (xyMin(1)+1) + (xyMax(1)-2 - (xyMin(1)+1))*rand();
            cy = (xyMin(2)+1) + (xyMax(2)-2 - (xyMin(2)+1))*rand();
            w  = 1.2 + 2.4*rand();
            h  = 1.0 + 2.2*rand();
            x1 = max(xyMin(1)+0.5, cx - w/2); x2 = min(xyMax(1)-0.5, cx + w/2);
            y1 = max(xyMin(2)+0.5, cy - h/2); y2 = min(xyMax(2)-0.5, cy + h/2);
        end
        extraPolys{i} = [x1 y1; x2 y1; x2 y2; x1 y2];
    end

    % Randomize GOAL
    goal_ok = false;
    for gtry=1:150
        gx = (xyMin(1)+1.0) + (xyMax(1)-2.0 - (xyMin(1)+1.0))*rand();
        gy = (xyMin(2)+1.0) + (xyMax(2)-2.0 - (xyMin(2)+1.0))*rand();
        goalXY = [gx, gy];

        % Provisional static occupancy
        occ0=false(mapRows,mapCols);
        polysToTest = [{blueWall, grayBox}, extraPolys];
        for p=1:numel(polysToTest)
            P=polysToTest{p};
            occ0 = occ0 | inpolygon(XC,YC,P(:,1),P(:,2));
        end
        se_static = strel('disk', max(1, ceil(safetyMargin / gridRes)), 0);
        occ0 = imdilate(occ0, se_static);

        rcG = world2grid(goalXY); rcG(1)=min(max(rcG(1),1),mapRows); rcG(2)=min(max(rcG(2),1),mapCols);
        rcS = world2grid(startXY); rcS(1)=min(max(rcS(1),1),mapRows); rcS(2)=min(max(rcS(2),1),mapCols);
        if ~occ0(rcG(1),rcG(2)) && ~occ0(rcS(1),rcS(2)) && norm(goalXY-startXY) > 5.0
            % quick static A* path check
            neighbors = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
            stepCost  = [ sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2) ];
            gScore=inf(mapRows,mapCols); fScore=inf(mapRows,mapCols);
            cameFrom=zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
            heur=@(r,c) hypot(double(r-rcG(1)),double(c-rcG(2)));
            gScore(rcS(1),rcS(2))=0; fScore(rcS(1),rcS(2))=heur(rcS(1),rcS(2)); openSet(rcS(1),rcS(2))=true;
            found=false; it=0;
            while it<6000
                it=it+1; ftmp=fScore; ftmp(~openSet)=inf;
                [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
                [r,c]=ind2sub(size(occ0),idxLin);
                if r==rcG(1) && c==rcG(2), found=true; break; end
                openSet(r,c)=false;
                for nb=1:8
                    rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
                    if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
                    if occ0(rr,cc), continue; end
                    if (abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1)
                        if occ0(r,cc) || occ0(rr,c), continue; end
                    end
                    tentative=gScore(r,c)+stepCost(nb);
                    if tentative < gScore(rr,cc)
                        cameFrom(rr,cc,:)=int32([r c]);
                        gScore(rr,cc)=tentative;
                        fScore(rr,cc)=tentative+heur(rr,cc);
                        openSet(rr,cc)=true;
                    end
                end
            end
            if found, goal_ok=true; break; end
        end
    end

    if goal_ok
        break;
    elseif attempt==maxTries
        warning('Proceeding without guaranteed static path (attempt %d).', maxTries);
    end
end

% Collect static polys & colors
staticPolys = [{blueWall, grayBox}, extraPolys];
polyColors  = [{[0 0 1]}, {[0.5 0.5 0.5]}];
while numel(polyColors) < numel(staticPolys)
    polyColors{end+1} = 0.25 + 0.5*rand(1,3); 
end

%% --- Dynamic obstacle (random) + wind ---
dyn_w = 3.0 + 0.8*rand(); dyn_h = 2.2 + 0.6*rand();
dcx = -2.5 + 5.0*rand(); dcy = -6.0 + 3.0*rand();
dynBase=[dcx-dyn_w/2 dcy-dyn_h/2;
         dcx+dyn_w/2 dcy-dyn_h/2;
         dcx+dyn_w/2 dcy+dyn_h/2;
         dcx-dyn_w/2 dcy+dyn_h/2];
dynAmp = max(1.2, min(3.0, 2.0 + 0.6*randn())); 
dynPer = max(6.0, min(10.0, 8.0 + 1.0*randn()));

% Optional second moving obstacle (40% chance)
USE_DYN2 = rand()<0.40;
if USE_DYN2
    dyn2_w = 2.0 + 1.2*rand(); dyn2_h = 1.8 + 1.0*rand();
    d2cx = -4 + 8*rand(); d2cy = -2 + 6*rand();
    dyn2Base=[d2cx-dyn2_w/2 d2cy-dyn2_h/2;
              d2cx+dyn2_w/2 d2cy-dyn2_h/2;
              d2cx+dyn2_w/2 d2cy+dyn2_h/2;
              d2cx-dyn2_w/2 d2cy+dyn2_h/2];
    dyn2Amp = 1.0 + 0.8*rand();  dyn2Per = 5.5 + 2.5*rand();
else
    dyn2Base=[]; dyn2Amp=0; dyn2Per=1;
end

wind_xy=[0.20 -0.10]; wind_sigma=0.04;

%% --- Control / planner params ---
dt=0.12; Tmax=130; Nsteps=round(Tmax/dt);
vmax=0.85; tolReach=0.30;

lookahead_m = 1.4; max_turn = pi; lowpass  = 0.18;
w_ml = 0.18;

dirs = deg2rad(0:45:315); ray_max=6; ray_step=gridRes*0.5;
shortcut_step = 0.08; min_clearance = 0.18; 

neighbors = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
stepCost  = [ sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2) ];
clear_pref = 0.85; clear_softmin = safetyMargin + 0.18; 

%% === FSM states ===
STATE_NORMAL      = 0;
STATE_CLIMB       = 1;
STATE_POSTREL     = 2;
STATE_LATERALCLR  = 3;
STATE_DESCENT     = 4;
state             = STATE_NORMAL;

% ---- UNSTICK ----
STATE_UNSTICK     = 5;
STUCK_VEL_EPS   = 0.04; 
STUCK_MOV_EPS   = 0.12; 
STUCK_WIN_T     = 2.4;
UNSTICK_T       = 2.1;
UNSTICK_POP     = 0.65;
unstick_until   = 0;
escape_dir      = [0 0];

% CLIMB/POSTREL/LATERAL CLEAR params
climb_band    = safetyMargin + 0.20;
climb_until_y = wall_top_y + (safetyMargin + 0.75);
release_hyst  = safetyMargin + 0.30;
hold_offset   = safetyMargin + 0.15;
climb_speed   = 0.85*vmax; min_climb_vy  = 0.60*climb_speed; x_pin=NaN;

POSTREL_T=1.2; POSTREL_PUSH=0.95;
LCLR_CLEAR_X = wall_thickness/2 + safetyMargin + 0.70;
LCLR_MAX_T   = 3.0; postrel_until=0; lclr_until=0; lat_sign=0;

% DESCENT 
DESC_SPEED = 0.95*vmax; DESC_MAX_T=6.0; DESC_KX=0.60;
desc_x_lock = 0; descent_until=0;

% Downward bias fallback
DOWNLOCK_ON = true; DOWNLOCK_GAIN = 0.78;

% Lip penalty band
lip_y_top=wall_top_y; lip_band_h=safetyMargin+0.35;
lip_x_min=x_left-(safetyMargin+0.10); lip_x_max=x_right+(safetyMargin+0.10);
lip_penalty_w=1.4;

% Corner-pop / sticky sliding
POP_THRESH=safetyMargin+0.25; POP_GAIN_MAX=0.58;
STICK_T=0.8; STICK_THRESH=0.18; stick_until=0; stick_sign=0;

% Dynamic obstacle inflation & pass latch
dyn_extra_margin=0.35;
r_dyn_pix = max(1, ceil((safetyMargin + dyn_extra_margin)/gridRes));
PASS_LATCH_T=2.0; pass_until=0; pass_dir=0;

%% --- State ---
USE_WIND=true; USE_DYN=true; USE_ML=true; LOG_CSV=true;
pos=startXY; vel=[0 0]; prev_u=[0 0];
traj=zeros(Nsteps,2); dmin_hist=nan(Nsteps,1); speed_hist=nan(Nsteps,1); t_hist=nan(Nsteps,1);

%% --- Video ---
vidW = 900; vidH = 900;
fig = figure('Color','w','Visible','on', ...
             'Units','pixels','Position',[100 100 vidW vidH], ...
             'Resize','off');
ax = axes('Parent',fig,'Units','pixels','Position',[60 60 vidW-120 vidH-120]);
axis(ax,'equal'); xlim(ax,[xyMin(1) xyMax(1)]); ylim(ax,[xyMin(2) xyMax(2)]);
set(ax,'LooseInset',[0 0 0 0]); box(ax,'on');

vOut = VideoWriter('week9_demo.mp4','MPEG-4');
vOut.FrameRate = round(1/dt);
open(vOut);
targetSize = [];

% Morphology SEs
se_static = strel('disk', max(1, ceil(safetyMargin / gridRes)), 0);
se_dyn  = strel('disk', r_dyn_pix, 0);

haveIsLineFree = exist('isLineFree','file')==2;

for k=1:Nsteps
    t=(k-1)*dt;

    % Dynamic obstacle(s) sweep
    dx = dynAmp*sin(2*pi*t/dynPer);
    dynPoly = dynBase+[dx 0; dx 0; dx 0; dx 0];

    if USE_DYN2
        dx2 = dyn2Amp*sin(2*pi*t/dyn2Per + pi/5);
        dyn2Poly = dyn2Base + [dx2 0; dx2 0; dx2 0; dx2 0];
    else
        dyn2Poly=[];
    end

    % Occupancy
    occ_static=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p};
        occ_static = occ_static | inpolygon(XC,YC,P(:,1),P(:,2));
    end
    occ_static = imdilate(occ_static, se_static);

    occ_dyn=false(mapRows,mapCols);
    if USE_DYN
        occ_dyn = occ_dyn | inpolygon(XC,YC,dynPoly(:,1),dynPoly(:,2));
        if ~isempty(dyn2Poly), occ_dyn = occ_dyn | inpolygon(XC,YC,dyn2Poly(:,1),dyn2Poly(:,2)); end
        occ_dyn = imdilate(occ_dyn, se_dyn);
    end
    occ = occ_static | occ_dyn;

    % Lip penalty mask
    lip_mask = (XC>=lip_x_min & XC<=lip_x_max & YC>=lip_y_top & YC<=lip_y_top+lip_band_h);

    % Distance map + gradients
    distMap = bwdist(occ)*gridRes; [gy,gx] = gradient(distMap, gridRes, gridRes);

    % ---------- Adaptive planning-only inflation ----------
    rcHere = world2grid(pos);
    rcHere(1)=min(max(rcHere(1),1),mapRows); rcHere(2)=min(max(rcHere(2),1),mapCols);
    d_here_now = distMap(rcHere(1),rcHere(2));

    plan_inflate_m = 0.0;
    if d_here_now < safetyMargin + 0.10
        plan_inflate_m = safetyMargin + 0.25;      
    elseif d_here_now < safetyMargin + 0.25
        plan_inflate_m = safetyMargin + 0.12;
    end
    if plan_inflate_m > 0
        r_plan_pix = max(1, ceil(plan_inflate_m / gridRes));
        occ_plan = imdilate(occ, strel('disk', r_plan_pix, 0));
    else
        occ_plan = occ;
    end

    % Bounds-checked occupancy query
    isOcc = @(p) ( ...
        p(1) < xyMin(1) || p(1) > xyMax(1) || ...
        p(2) < xyMin(2) || p(2) > xyMax(2) || ...
        occ( ...
            min(max(round((p(2)-xyMin(2))/gridRes)+1,1), mapRows), ...
            min(max(round((p(1)-xyMin(1))/gridRes)+1,1), mapCols) ...
        ) ...
    );

    % dmin scan
    dmin = ray_max;
    for kk=1:8
        ddir=[cos(dirs(kk)), sin(dirs(kk))]; s=0;
        while s<=ray_max
            q=pos+s*ddir;
            rr=round((q(2)-xyMin(2))/gridRes)+1; cc=round((q(1)-xyMin(1))/gridRes)+1;
            if rr<1||rr>mapRows||cc<1||cc>mapCols || occ(min(max(rr,1),mapRows),min(max(cc,1),mapCols))
                if s<dmin, dmin=s; end, break;
            end
            s=s+ray_step;
        end
    end

    % ------------ A* with lip penalty ------------
    sRC=world2grid(pos); gRC=world2grid(goalXY);
    sRC(1)=min(max(sRC(1),1),mapRows); sRC(2)=min(max(sRC(2),1),mapCols);
    gRC(1)=min(max(gRC(1),1),mapRows); gRC(2)=min(max(gRC(2),1),mapCols);

    if occ_plan(gRC(1),gRC(2))
        for rad=1:5
            rr=max(1,gRC(1)-rad):min(mapRows,gRC(1)+rad);
            cc=max(1,gRC(2)-rad):min(mapCols,gRC(2)+rad);
            [RR,CC]=ndgrid(rr,cc); ring=[RR(:),CC(:)];
            free = ~occ_plan(sub2ind(size(occ_plan), ring(:,1), ring(:,2)));
            if any(free), gRC = ring(find(free,1,'first'),:); goalXY=grid2world(gRC); break; end
        end
    end

    if occ_plan(sRC(1),sRC(2))
        for rad=1:3
            rr=max(1,sRC(1)-rad):min(mapRows,sRC(1)+rad);
            cc=max(1,sRC(2)-rad):min(mapCols,sRC(2)+rad);
            [RR,CC]=ndgrid(rr,cc); ring=[RR(:),CC(:)];
            free = ~occ_plan(sub2ind(size(occ_plan), ring(:,1), ring(:,2)));
            if any(free), sRC = ring(find(free,1,'first'),:); break; end
        end
    end

    gScore=inf(mapRows,mapCols); fScore=inf(mapRows,mapCols);
    cameFrom=zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
    gScore(sRC(1),sRC(2))=0; heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
    fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;

    it=0; foundPath=false;
    while it<7000
        it=it+1; ftmp=fScore; ftmp(~openSet)=inf;
        [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
        [r,c]=ind2sub(size(occ_plan),idxLin);
        if r==gRC(1) && c==gRC(2)
            pathRC=[r c];
            while ~(r==sRC(1) && c==sRC(2))
                prev=double(squeeze(cameFrom(r,c,:))'); if all(prev==0), break; end
                r=prev(1); c=prev(2); pathRC(end+1,:)=[r c];
            end
            pathRC=flipud(pathRC); foundPath=true; break
        end
        openSet(r,c)=false;
        for nb=1:8
            rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
            if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
            if occ_plan(rr,cc), continue; end
            if (abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1)
                if occ_plan(r,cc) || occ_plan(rr,c), continue; end
            end
            tentative=gScore(r,c)+stepCost(nb);
            dclr=distMap(rr,cc); extra=0;
            if dclr<clear_softmin, extra = extra + clear_pref*(1/max(dclr,1e-3) - 1/clear_softmin); end
            if lip_mask(rr,cc),  extra = extra + lip_penalty_w; end
            if tentative + extra < gScore(rr,cc) + 1e-12
                cameFrom(rr,cc,:)=int32([r c]);
                gScore(rr,cc)=tentative;
                fScore(rr,cc)=tentative+heur(rr,cc)+extra;
                openSet(rr,cc)=true;
            end
        end
    end
    if foundPath, pathXY=grid2world(pathRC); else, pathXY=[pos; goalXY]; end

    % ------------ String-pulling ------------
    sp=pos; smoothPath=sp; idx=2;
    while idx<=size(pathXY,1)
        j=idx; best_j=idx;
        while j<=size(pathXY,1)
            pA=sp; pB=pathXY(j,:); seg_free=true; seg_len=norm(pB-pA);
            if seg_len>0
                nsteps=max(2,ceil(seg_len/shortcut_step));
                for ss=1:nsteps
                    q=pA+(ss/nsteps)*(pB-pA);
                    if isOcc(q) || isOcc(q+[min_clearance 0]) || isOcc(q+[-min_clearance 0]) || ...
                                   isOcc(q+[0 min_clearance]) || isOcc(q+[0 -min_clearance])
                        seg_free=false; break;
                    end
                end
            end
            if seg_free, best_j=j; j=j+1; else, break; end
        end
        sp=pathXY(best_j,:); smoothPath(end+1,:)=sp; idx=best_j+1;
    end

    % ------------ V plan ------------
    v_plan=[0 0];
    if size(smoothPath,1)>=2
        acc=0; ii=2; target_wp=smoothPath(min(2,size(smoothPath,1)),:);
        while ii<=size(smoothPath,1)
            seg=norm(smoothPath(ii,:)-smoothPath(ii-1,:));
            if acc+seg>=lookahead_m
                need=lookahead_m-acc; dirseg=(smoothPath(ii,:)-smoothPath(ii-1,:))/max(seg,1e-6);
                target_wp=smoothPath(ii-1,:)+need*dirseg; break;
            else
                acc=acc+seg; target_wp=smoothPath(ii,:); ii=ii+1;
            end
        end
        dirp=target_wp-pos; ndp=norm(dirp); if ndp>1e-6, v_plan=dirp/ndp; end
    else
        dirp=goalXY-pos; ndp=norm(dirp); if ndp>1e-6, v_plan=dirp/ndp; end
    end

    % ------------ ML nudge ------------
    dist8=zeros(8,1);
    for kk=1:8
        ddir=[cos(dirs(kk)), sin(dirs(kk))]; s=0; d=ray_max;
        while s<=ray_max
            q=pos+s*ddir;
            rr=round((q(2)-xyMin(2))/gridRes)+1; cc=round((q(1)-xyMin(1))/gridRes)+1;
            if rr<1||rr>mapRows||cc<1||cc>mapCols || occ(min(max(rr,1),mapRows),min(max(cc,1),mapCols))
                d=s; break;
            end
            s=s+ray_step;
        end
        dist8(kk)=d;
    end
    rel_goal=[goalXY-pos,0]; v3=[vel,0];
    feat=[rel_goal, v3, wind_xy, dist8']; fx=feat(keep_cols);
    yhat=[fx,1]*W_best; v_ml=yhat(1:2);

    % ---- Deadlock detector -> enter UNSTICK ----
    winN = max(1, ceil(STUCK_WIN_T/dt));
    if k > winN && any(traj(max(1,k-winN),:) ~= 0)
        net_mov = norm(pos - traj(k-winN, :));
    else
        net_mov = inf;
    end
    is_slow  = (norm(vel) < STUCK_VEL_EPS);
    is_stuck = (net_mov < STUCK_MOV_EPS) || is_slow;

    if (state ~= STATE_CLIMB) && (state ~= STATE_DESCENT)
        if is_stuck && (t >= unstick_until) && k > 2
            % Choose escape direction = blend of normal + goal
            rc_e = world2grid(pos);
            rc_e(1)=min(max(rc_e(1),1),mapRows); rc_e(2)=min(max(rc_e(2),1),mapCols);
            n_e = [gx(rc_e(1),rc_e(2)), gy(rc_e(1),rc_e(2))];
            if norm(n_e) < 1e-9, n_e = [0 0]; else, n_e = n_e / norm(n_e); end
            to_goal = goalXY - pos; 
            if norm(to_goal) < 1e-9, gdir = [1 0]; else, gdir = to_goal / norm(to_goal); end
            esc = 0.60*n_e + 0.40*gdir;

            if norm(esc) < 1e-6
                bestd = -inf; esc = [1,0];
                for kk2=1:8
                    ddir=[cos(dirs(kk2)), sin(dirs(kk2))]; s=0; d=ray_max;
                    while s<=ray_max
                        q=pos+s*ddir;
                        rr=round((q(2)-xyMin(2))/gridRes)+1; cc=round((q(1)-xyMin(1))/gridRes)+1;
                        if rr<1||rr>mapRows||cc<1||cc>mapCols || occ(min(max(rr,1),mapRows),min(max(cc,1),mapCols))
                            d=s; break;
                        end
                        s=s+ray_step;
                    end
                    if d>bestd, bestd=d; esc=ddir; end
                end
            end
            escape_dir = esc / max(1e-9,norm(esc));
            unstick_until = t + UNSTICK_T;
            state = STATE_UNSTICK; prev_u=[0 0];
        end
    end

    %% ------------ FSM transitions ------------
    near_wall_by_x = (pos(1)>=x_left - (safetyMargin+0.20)) && (pos(1)<=x_right + (safetyMargin+0.20));
    below_top      = (pos(2) < climb_until_y);
    above_release  = (pos(2) > climb_until_y + (safetyMargin + 0.30));

    switch state
        case STATE_NORMAL
            if near_wall_by_x && below_top
                if pos(1) <= x_wall_center, x_pin = x_left  - (safetyMargin + 0.15);
                else, x_pin = x_right + (safetyMargin + 0.15);
                end
                state = STATE_CLIMB;
            end
        case STATE_CLIMB
            if above_release
                lat_sign = sign(goalXY(1) - x_wall_center); if lat_sign==0, lat_sign=1; end
                postrel_until = t + POSTREL_T;
                lclr_until    = t + LCLR_MAX_T;
                state = STATE_POSTREL; prev_u=[0 0];
            end
        case STATE_POSTREL
            if t >= postrel_until, state = STATE_LATERALCLR; end
        case STATE_LATERALCLR
            if (abs(pos(1) - x_wall_center) >= LCLR_CLEAR_X) || (t >= lclr_until)
                state = STATE_NORMAL; prev_u=[0 0];
            end
        case STATE_DESCENT
            if (pos(2) <= goalXY(2)+0.15) || (t >= descent_until)
                state = STATE_NORMAL; prev_u=[0 0];
            end
        case STATE_UNSTICK
            if t >= unstick_until
                state = STATE_NORMAL; prev_u=[0 0]; pass_dir=0; stick_until=0;
            end
    end

    %% ------------ DESCENT engage ------------
    if state==STATE_NORMAL
        above_goal = pos(2) > (goalXY(2) + 0.25);
        clear_of_slab = abs(pos(1) - x_wall_center) >= (wall_thickness/2 + safetyMargin*0.50);
        if above_goal && clear_of_slab
            okH=false; okV=false;
            if haveIsLineFree
                pH = [goalXY(1), pos(2)];
                pV = [goalXY(1), goalXY(2)];
                okH = isLineFree(pos, pH, @(q) isOcc(q), gridRes*0.25, min_clearance);
                okV = isLineFree(pH,  pV, @(q) isOcc(q), gridRes*0.25, min_clearance);
            end
            if (~haveIsLineFree && abs(pos(1)-goalXY(1))<1.0) || (okH && okV)
                desc_x_lock   = goalXY(1);
                descent_until = t + DESC_MAX_T;
                state = STATE_DESCENT; prev_u=[0 0];
            end
        end
    end

    %% ------------ Command composition ------------
    u_nom = v_plan + w_ml*v_ml;
    above_wall = pos(2) > lip_y_top + 0.05;
    goal_below = goalXY(2) < pos(2) - 0.1;
    if DOWNLOCK_ON && state==STATE_NORMAL && above_wall && goal_below
        x_correction = max(-1,min(1, (goalXY(1)-pos(1))/2.0 ));
        down_vec = [0, -1];
        u_nom = 0.78*(down_vec + [0.35*x_correction, 0]) + 0.22*u_nom;
    end

    % Dynamic obstacle side-pass latch
    dynBoxList = [];
    if ~isempty(dynPoly)
        dynBoxList(end+1,:) = [min(dynPoly(:,1)) max(dynPoly(:,1)) min(dynPoly(:,2)) max(dynPoly(:,2))];
    end
    if ~isempty(dyn2Poly)
        dynBoxList(end+1,:) = [min(dyn2Poly(:,1)) max(dyn2Poly(:,1)) min(dyn2Poly(:,2)) max(dyn2Poly(:,2))];
    end
    expand = dyn_extra_margin + safetyMargin*0.6;
    for bi=1:size(dynBoxList,1)
        x1=dynBoxList(bi,1)-expand; x2=dynBoxList(bi,2)+expand;
        y1=dynBoxList(bi,3)-expand; y2=dynBoxList(bi,4)+expand;
        line_to_goal_crosses = ...
            (min(pos(1),goalXY(1)) <= x2) && (max(pos(1),goalXY(1)) >= x1) && ...
            (min(pos(2),goalXY(2)) <= y2) && (max(pos(2),goalXY(2)) >= y1);
        near_dyn_band = (pos(2) > y1 - 1.0) && (pos(2) < y2 + 1.0);
        if USE_DYN && line_to_goal_crosses && near_dyn_band && (t > pass_until) && state~=STATE_CLIMB
            sample_off = 1.0;
            rcL = world2grid([pos(1)-sample_off, pos(2)]);
            rcR = world2grid([pos(1)+sample_off, pos(2)]);
            rcL(1)=min(max(rcL(1),1),mapRows); rcL(2)=min(max(rcL(2),1),mapCols);
            rcR(1)=min(max(rcR(1),1),mapRows); rcR(2)=min(max(rcR(2),1),mapCols);
            dL = distMap(rcL(1),rcL(2)); dR = distMap(rcR(1),rcR(2));
            pass_dir = (dL > dR)*(-1) + (dR >= dL)*(+1);
            pass_until = t + PASS_LATCH_T;
        end
    end
    if pass_dir ~= 0 && state==STATE_NORMAL
        u_nom = 0.55*u_nom + 0.45*[pass_dir, 0];
    end

    % Corridor tangent & corner-pop
    rc = world2grid(pos);
    rc(1)=min(max(rc(1),1),mapRows); rc(2)=min(max(rc(2),1),mapCols);
    d_here = distMap(rc(1),rc(2));
    n_est = [gx(rc(1),rc(2)), gy(rc(1),rc(2))]; n_est = n_est / max(norm(n_est),1e-9);
    if state==STATE_NORMAL && d_here < POP_THRESH
        gain = POP_GAIN_MAX * (POP_THRESH - d_here) / POP_THRESH;
        u_nom = u_nom + gain * n_est;
    end
    % Tangent bias along corridor when close to obstacles
    if state==STATE_NORMAL && d_here < (safetyMargin + 0.22)
        t_est = [ -n_est(2), n_est(1) ];
        to_goal_dir = (goalXY - pos); 
        ng = norm(to_goal_dir); if ng>1e-9, to_goal_dir = to_goal_dir/ng; else, to_goal_dir = [1 0]; end
        tang_sign = sign(dot(t_est, to_goal_dir)); if tang_sign==0, tang_sign=1; end
        t_push = tang_sign * t_est;
        u_nom = 0.60*u_nom + 0.40*t_push;
    end

    % State-specific actuation
    if state==STATE_CLIMB
        u = [0, max(climb_speed, min_climb_vy)]; prev_u=u;

    elseif state==STATE_POSTREL
        to_goal = goalXY - pos; dg = max(norm(to_goal),1e-6); goal_dir = to_goal/dg;
        u = POSTREL_PUSH * [lat_sign, 0] + 0.30*goal_dir; prev_u=u;

    elseif state==STATE_LATERALCLR
        to_goal = goalXY - pos; dg = max(norm(to_goal),1e-6); goal_dir = to_goal/dg;
        u = 0.85 * [lat_sign, 0] + 0.30*goal_dir; prev_u=u;

    elseif state==STATE_DESCENT
        x_err = desc_x_lock - pos(1);
        u = [DESC_KX*max(-1,min(1,x_err)), -DESC_SPEED]; prev_u=u;

    elseif state==STATE_UNSTICK
        t_est = [ -n_est(2), n_est(1) ];
        jitter = 0.10 * (2*rand(1,2)-1);                 
        goal_dir = goalXY - pos; ng = norm(goal_dir); if ng>1e-9, goal_dir=goal_dir/ng; else, goal_dir=[1 0]; end
        % If escape points into obstacle normal, favor tangent
        esc_blend = escape_dir;
        if dot(escape_dir, n_est) > 0.6
            esc_blend = 0.5*escape_dir + 0.5*t_est*sign(dot(t_est,goal_dir));
            esc_blend = esc_blend / max(norm(esc_blend),1e-9);
        end
        u = 0.72*esc_blend + (UNSTICK_POP+0.10)*n_est + 0.18*goal_dir + jitter;
        prev_u = u;

    else
        if norm(u_nom)>1e-6 && norm(prev_u)>1e-6
            hu=u_nom/norm(u_nom); hp=prev_u/norm(prev_u);
            ang=atan2(hu(2),hu(1))-atan2(hp(2),hp(1)); ang=atan2(sin(ang),cos(ang));
            ang_clamped=max(min(ang,max_turn),-max_turn);
            if abs(ang_clamped)<abs(ang)
                speed=norm(u_nom); target_heading=atan2(hp(2),hp(1))+ang_clamped;
                u_nom = speed*[cos(target_heading), sin(target_heading)];
            end
        end
        u = (1-lowpass)*u_nom + lowpass*prev_u; prev_u=u;
    end

    % Speed scaling
    ttc_like=max(min(dmin/ray_max,1),0);
    base_floor = any(state==[STATE_POSTREL,STATE_LATERALCLR,STATE_DESCENT,STATE_UNSTICK]) * 0.70 + ...
                 ~any(state==[STATE_POSTREL,STATE_LATERALCLR,STATE_DESCENT,STATE_UNSTICK]) * 0.55;
    speed_scale = max(base_floor, base_floor + (1-base_floor)*ttc_like);
    spd=norm(u); if spd>vmax*speed_scale, u=(vmax*speed_scale/spd)*u; end
    if state==STATE_CLIMB, u(1)=0; u(2)=max(u(2),min_climb_vy); end

    % ------------ Integrate ------------
    wind_term=[0 0]; if USE_WIND, wind_term = wind_xy + wind_sigma*randn(1,2); if state==STATE_CLIMB, wind_term(1)=0; end, end
    v_apply = u + wind_term;
    if state==STATE_CLIMB && v_apply(2)<min_climb_vy, v_apply(2)=min_climb_vy; end
    if state==STATE_CLIMB, v_apply(1)=0; end

    tryStep = v_apply*dt; seg_len=norm(tryStep);
    max_step=0.30*gridRes; Nsub=max(1,ceil(seg_len/max_step)); subStep=tryStep/Nsub;

    for ss=1:Nsub
        p_try = pos + subStep;
        if state==STATE_CLIMB && pos(2) < climb_until_y, p_try(1)=x_pin; end

        hit = isOcc(p_try) || isOcc(p_try+[min_clearance 0]) || ...
              isOcc(p_try+[-min_clearance 0]) || isOcc(p_try+[0 min_clearance]) || ...
              isOcc(p_try+[0 -min_clearance]);

        if hit
            if state==STATE_CLIMB
                p_try(1)=x_pin; p_try(2)=pos(2)+max(gridRes*0.25, subStep(2));
                if isOcc(p_try), p_try(2)=pos(2)+gridRes*0.25; end

            elseif state==STATE_DESCENT
                % LoS blocked mid-descent: bail to NORMAL and re-plan
                state = STATE_NORMAL; prev_u=[0 0]; break;

            else
                rc2=world2grid(pos);
                rc2(1)=min(max(rc2(1),1),mapRows); rc2(2)=min(max(rc2(2),1),mapCols);
                n2=[gx(rc2(1),rc2(2)), gy(rc2(1),rc2(2))];
                if norm(n2)<1e-6, n2=[-v_apply(2), v_apply(1)]; end
                n2=n2/max(norm(n2),1e-6); t2=[n2(2), -n2(1)];

                if state==STATE_NORMAL || state==STATE_UNSTICK
                    if t < stick_until
                        if sign(t2(1)) ~= stick_sign, t2 = -t2; end
                    else
                        if dot(t2, v_apply) < 0, t2 = -t2; end
                    end
                    if abs(t2(1)) > STICK_THRESH
                        stick_sign = sign(t2(1)); stick_until = t + STICK_T;
                    end
                end

                if any(state==[STATE_POSTREL,STATE_LATERALCLR])
                    to_goal = goalXY - pos; dg = max(norm(to_goal),1e-6); goal_dir = to_goal/dg;
                    v_apply = 0.85*[lat_sign,0] + 0.15*goal_dir;
                else
                    % stronger tangent: reduce bounce while sliding
                    v_apply = 0.88*dot(v_apply,t2)*t2 + 0.08*v_plan + 0.04*n2;
                end

                tryStep = v_apply * (dt*(1 - (ss-1)/Nsub));
                seg_len2=norm(tryStep); Nsub2=max(1,ceil(seg_len2/max_step));
                subStep=tryStep/Nsub2; Nsub=ss-1+Nsub2; continue
            end
        end
        pos = p_try;
    end
    vel = v_apply;

    % Log & goal check
    traj(k,:)=pos; dmin_hist(k)=dmin; speed_hist(k)=norm(vel); t_hist(k)=t;
    if norm(pos-goalXY)<=tolReach, break; end

    % ===== Draw (fixed-size figure/axes + robust legend) =====
    cla(ax); hold(ax,'on');
    axis(ax,'equal'); xlim(ax,[xyMin(1) xyMax(1)]); ylim(ax,[xyMin(2) xyMax(2)]);

    Lh = []; Ll = strings(0,1);

    % Static objects
    hBlue = patch('Parent',ax, 'XData',staticPolys{1}(:,1), 'YData',staticPolys{1}(:,2), ...
                  'FaceColor',[0 0 1],'FaceAlpha',0.5,'EdgeColor','k');
    Lh(end+1) = hBlue; Ll(end+1) = "blue wall";

    hGray = patch('Parent',ax, 'XData',staticPolys{2}(:,1), 'YData',staticPolys{2}(:,2), ...
                  'FaceColor',[0.5 0.5 0.5],'FaceAlpha',0.5,'EdgeColor','k');
    Lh(end+1) = hGray; Ll(end+1) = "gray box";

    % Extra statics (representative)
    hExtraRep = [];
    for p=3:numel(staticPolys)
        P = staticPolys{p};
        h = patch('Parent',ax, 'XData',P(:,1),'YData',P(:,2), ...
                  'FaceColor',polyColors{p},'FaceAlpha',0.5,'EdgeColor','k');
        if isempty(hExtraRep), hExtraRep = h; end
    end
    if ~isempty(hExtraRep), Lh(end+1) = hExtraRep; Ll(end+1) = "extra static objs"; end

    % Dynamic obstacles
    hDyn1 = patch('Parent',ax, 'XData',dynPoly(:,1),'YData',dynPoly(:,2), ...
                  'FaceColor',[.85 .4 .1],'FaceAlpha',0.6,'EdgeColor','k');
    Lh(end+1) = hDyn1; Ll(end+1) = "dynamic 1";

    if ~isempty(dyn2Poly)
        hDyn2 = patch('Parent',ax, 'XData',dyn2Poly(:,1),'YData',dyn2Poly(:,2), ...
                      'FaceColor',[.2 .7 .2],'FaceAlpha',0.55,'EdgeColor','k');
        Lh(end+1) = hDyn2; Ll(end+1) = "dynamic 2";
    end

    % Lip band
    hLip = patch('Parent',ax, ...
                 'XData',[lip_x_min lip_x_max lip_x_max lip_x_min], ...
                 'YData',[lip_y_top lip_y_top lip_y_top+lip_band_h lip_y_top+lip_band_h], ...
                 'FaceColor',[0.6 0.8 1.0], 'FaceAlpha',0.15, 'EdgeColor','none');
    Lh(end+1) = hLip; Ll(end+1) = "lip band";

    % Start/Goal
    hStart = plot(ax, startXY(1),startXY(2),'go','MarkerFaceColor','g');
    Lh(end+1) = hStart; Ll(end+1) = "start";

    hGoal = plot(ax, goalXY(1),goalXY(2),'rx','LineWidth',2,'MarkerSize',10);
    Lh(end+1) = hGoal; Ll(end+1) = "goal";

    % Paths & trajectory
    if exist('pathXY','var') && size(pathXY,1)>1
        hA = plot(ax, pathXY(:,1), pathXY(:,2),'m:','LineWidth',1.1);
        Lh(end+1) = hA; Ll(end+1) = "A* path";
    end
    if exist('smoothPath','var') && size(smoothPath,1)>1
        hS = plot(ax, smoothPath(:,1), smoothPath(:,2),'m-','LineWidth',1.8);
        Lh(end+1) = hS; Ll(end+1) = "smoothed path";
    end
    hT = plot(ax, traj(1:k,1),traj(1:k,2),'r-','LineWidth',2);
    Lh(end+1) = hT; Ll(end+1) = "trajectory";

    % Control arrow
    hU = quiver(ax, pos(1),pos(2),u(1),u(2),0,'k');
    Lh(end+1) = hU; Ll(end+1) = "u (cmd)";

    % Pin line during climb
    if state==STATE_CLIMB
        xline(ax, x_pin,'--k');
    end

    % Legend 
    isValid = arrayfun(@isgraphics, Lh);
    leg = legend(ax, Lh(isValid), cellstr(Ll(isValid)), ...
                 'Location','northwest', 'AutoUpdate','off');
    set(leg,'Units','pixels');

    title(ax, sprintf('t=%.1fs  dmin=%.2f  speed=%.2f  state=%d', t, dmin, norm(vel), state));
    drawnow;

    % ---- Capture & normalize frame size to first frame ----
    frame = getframe(fig);
    if isempty(targetSize)
        targetSize = size(frame.cdata);   % lock on first frame
    else
        if ~isequal(size(frame.cdata), targetSize)
            frame.cdata = imresize(frame.cdata, [targetSize(1) targetSize(2)]);
        end
    end
    writeVideo(vOut, frame);
end

close(vOut);
saveas(fig,'week9_final_path.png');

% CSV dump
if LOG_CSV
    valid = ~isnan(t_hist);
    T = table(t_hist(valid), traj(valid,1), traj(valid,2), speed_hist(valid), dmin_hist(valid), ...
        'VariableNames', {'t','x','y','speed','dmin'});
    writetable(T,'week9_metrics.csv');
end
