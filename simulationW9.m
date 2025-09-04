%% Week 9 — Integration with Simulation

clear; clc; close all; clear fig; rng(77);

%% --- Load Week 6 dataset & ridge policy ---
S = load('week6_dataset.mat'); % features/actions/split
X  = S.features;  Y  = S.actions;
itr = S.split.train(:);  iva = S.split.val(:);
Xtr = X(itr,:);   Ytr = Y(itr,:);
Xva = X(iva,:);   Yva = Y(iva,:);

std_tr    = std(Xtr,0,1);
keep_cols = std_tr > 1e-8;
Xtr = Xtr(:,keep_cols);  Xva = Xva(:,keep_cols);
Xtr_b=[Xtr,ones(size(Xtr,1),1)]; Xva_b=[Xva,ones(size(Xva,1),1)];
XtX=Xtr_b'*Xtr_b; XtY=Xtr_b'*Ytr; I=eye(size(Xtr_b,2)); I(end,end)=0;
lambdas=logspace(-4,2,20); bestVa=inf; W_best=[];
for lam=lambdas
    W=(XtX+lam*I)\XtY;
    va_rmse=sqrt(mean(sum((Xva_b*W-Yva).^2,2)));
    if va_rmse<bestVa, bestVa=va_rmse; W_best=W; end
end

%% --- World & obstacles ---
gridRes=0.20; safetyMargin=0.60;
xyMin=[-10 -10]; xyMax=[10 10];
startXY=[-8,-8]; goalXY=[8,2];

% Blue wall (vertical slab) — top at y=5
x_wall_center = 0.60; wall_thickness = 0.40;
x_left  = x_wall_center - wall_thickness/2;
x_right = x_wall_center + wall_thickness/2;
blueWall = [x_left -10; x_right -10; x_right 5.0; x_left 5.0];

% Gray box (true box)
grayBox  = [4 4; 7 4; 7 7; 4 7]; box_ymax=7;

staticPolys = {blueWall, grayBox};
polyColors  = {'b', [0.5 0.5 0.5]};

% Grid
xv=xyMin(1):gridRes:xyMax(1); yv=xyMin(2):gridRes:xyMax(2);
[XC,YC]=meshgrid(xv,yv); [mapRows,mapCols]=size(XC);

% Converters
world2grid=@(xy)[ round((xy(2)-xyMin(2))/gridRes)+1, round((xy(1)-xyMin(1))/gridRes)+1 ];
grid2world=@(rc)[ xyMin(1)+(rc(:,2)-1)*gridRes, xyMin(2)+(rc(:,1)-1)*gridRes ];

%% --- Dynamic obstacle + wind ---
dynBase=[-1.8 -6.2; 1.8 -6.2; 1.8 -3.8; -1.8 -3.8];
dynAmp = 2.0;  dynPer = 8.0;
wind_xy=[0.20 -0.10]; wind_sigma=0.04;

%% --- Control / planner params ---
dt=0.12; Tmax=110; Nsteps=round(Tmax/dt);
vmax=0.85; tolReach=0.30;

lookahead_m = 1.4; max_turn = pi; lowpass  = 0.18;
w_ml = 0.18;

dirs = deg2rad(0:45:315); ray_max=6; ray_step=gridRes*0.5;

shortcut_step = 0.10; min_clearance = 0.12;

neighbors = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
stepCost  = [ sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2) ];

clear_pref    = 0.55; clear_softmin = safetyMargin + 0.10;

%% === FSM (CLIMB / POSTREL / LATERALCLR / DESCENT) ===
STATE_NORMAL      = 0;
STATE_CLIMB       = 1;
STATE_POSTREL     = 2;
STATE_LATERALCLR  = 3;
STATE_DESCENT     = 4;
state             = STATE_NORMAL;

% climb
climb_band    = safetyMargin + 0.20;
climb_until_y = 6.60; release_hyst  = safetyMargin + 0.30;
hold_offset   = safetyMargin + 0.15;
climb_speed   = 0.85*vmax; min_climb_vy  = 0.60*climb_speed; x_pin=NaN;

% post-release & lateral-clear
POSTREL_T=1.2; POSTREL_PUSH=0.95;
LCLR_CLEAR_X = wall_thickness/2 + safetyMargin + 0.70;
LCLR_MAX_T   = 3.0; postrel_until=0; lclr_until=0; lat_sign=0;

% descent (with fallback)
DESC_SPEED = 0.95*vmax; DESC_MAX_T=6.0; DESC_KX=0.60;
desc_x_lock = goalXY(1); descent_until=0;

% “downward intent” bias to prevent up-hover when above wall & goal below
DOWNLOCK_ON = true; DOWNLOCK_GAIN = 0.78; % strong downward pull when applicable

%% lip penalty (discourage skimming)
lip_y_top=5.0; lip_band_h=safetyMargin+0.35;
lip_x_min=x_left-(safetyMargin+0.10); lip_x_max=x_right+(safetyMargin+0.10);
lip_penalty_w=1.4;

%% corner-pop / sticky sliding
POP_THRESH=safetyMargin+0.25; POP_GAIN_MAX=0.58;
STICK_T=0.8; STICK_THRESH=0.18; stick_until=0; stick_sign=0;

%% dynamic obstacle inflation & pass latch
dyn_extra_margin=0.35;
r_dyn_pix = max(1, ceil((safetyMargin + dyn_extra_margin)/gridRes));
PASS_LATCH_T=2.0; pass_until=0; pass_dir=0;

%% state vars
USE_WIND=true; USE_DYN=true; USE_ML=true; LOG_CSV=true;
pos=startXY; vel=[0 0]; prev_u=[0 0];
traj=zeros(Nsteps,2); dmin_hist=nan(Nsteps,1); speed_hist=nan(Nsteps,1); t_hist=nan(Nsteps,1);

%% video
fig=figure('Color','w','Visible','on'); try, set(fig,'Position',[100 100 860 860]); end
vOut=VideoWriter('week9_demo.mp4','MPEG-4'); vOut.FrameRate=round(1/dt); open(vOut);

% morphology
r_pix = max(1, ceil(safetyMargin / gridRes)); se_static = strel('disk', r_pix, 0);
se_dyn  = strel('disk', r_dyn_pix, 0);

% helper availability
haveIsLineFree = exist('isLineFree','file')==2;

for k=1:Nsteps
    t=(k-1)*dt;

    % dynamic obstacle
    dx=dynAmp*sin(2*pi*t/dynPer);
    dynPoly=dynBase+[dx 0;dx 0;dx 0;dx 0];

    % occupancy (static + dyn)
    occ_static=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p};
        occ_static = occ_static | inpolygon(XC,YC,P(:,1),P(:,2));
    end
    occ_static = imdilate(occ_static, se_static);
    occ_dyn=false(mapRows,mapCols);
    if USE_DYN
        occ_dyn = inpolygon(XC,YC,dynPoly(:,1),dynPoly(:,2));
        occ_dyn = imdilate(occ_dyn, se_dyn);
    end
    occ = occ_static | occ_dyn;

    % lip penalty mask
    lip_mask = (XC>=lip_x_min & XC<=lip_x_max & YC>=lip_y_top & YC<=lip_y_top+lip_band_h);

    % distance map & grads
    distMap = bwdist(occ)*gridRes; [gy,gx] = gradient(distMap, gridRes, gridRes);

    % bounds-checked occupancy
    isOcc = @(p) ( ...
        p(1) < xyMin(1) || p(1) > xyMax(1) || ...
        p(2) < xyMin(2) || p(2) > xyMax(2) || ...
        occ( ...
            min(max(round((p(2)-xyMin(2))/gridRes)+1,1), mapRows), ...
            min(max(round((p(1)-xyMin(1))/gridRes)+1,1), mapCols) ...
        ) ...
    );

    % dmin for speed scaling
    dmin = ray_max;
    for kk=1:8
        dir=[cos(dirs(kk)), sin(dirs(kk))]; s=0;
        while s<=ray_max
            q=pos+s*dir;
            rr=round((q(2)-xyMin(2))/gridRes)+1; cc=round((q(1)-xyMin(1))/gridRes)+1;
            if rr<1||rr>mapRows||cc<1||cc>mapCols || occ(min(max(rr,1),mapRows),min(max(cc,1),mapCols))
                if s<dmin, dmin=s; end, break;
            end
            s=s+ray_step;
        end
    end

    % --- A* with lip penalty ---
    sRC=world2grid(pos); gRC=world2grid(goalXY);
    sRC(1)=min(max(sRC(1),1),mapRows); sRC(2)=min(max(sRC(2),1),mapCols);
    gRC(1)=min(max(gRC(1),1),mapRows); gRC(2)=min(max(gRC(2),1),mapCols);

    if occ(gRC(1),gRC(2))
        for rad=1:5
            rr=max(1,gRC(1)-rad):min(mapRows,gRC(1)+rad);
            cc=max(1,gRC(2)-rad):min(mapCols,gRC(2)+rad);
            [RR,CC]=ndgrid(rr,cc); ring=[RR(:),CC(:)];
            free = ~occ(sub2ind(size(occ), ring(:,1), ring(:,2)));
            if any(free), gRC = ring(find(free,1,'first'),:); goalXY=grid2world(gRC); break; end
        end
    end

    if occ(sRC(1),sRC(2))
        for rad=1:3
            rr=max(1,sRC(1)-rad):min(mapRows,sRC(1)+rad);
            cc=max(1,sRC(2)-rad):min(mapCols,sRC(2)+rad);
            [RR,CC]=ndgrid(rr,cc); ring=[RR(:),CC(:)];
            free = ~occ(sub2ind(size(occ), ring(:,1), ring(:,2)));
            if any(free), sRC = ring(find(free,1,'first'),:); break; end
        end
    end

    gScore=inf(mapRows,mapCols); fScore=inf(mapRows,mapCols);
    cameFrom=zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
    gScore(sRC(1),sRC(2))=0; heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
    fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;

    it=0; foundPath=false;
    while it<6000
        it=it+1; ftmp=fScore; ftmp(~openSet)=inf;
        [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
        [r,c]=ind2sub(size(occ),idxLin);
        if r==gRC(1) && c==gRC(2)
            pathRC=[r c];
            while ~(r==sRC(1) && c==sRC(2))
                prev=double(squeeze(cameFrom(r,c,:))'); if all(prev==0), break; end
                r=prev(1); c=prev(2); pathRC(end+1,:)=[r c]; %#ok<AGROW>
            end
            pathRC=flipud(pathRC); foundPath=true; break
        end
        openSet(r,c)=false;
        for nb=1:8
            rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
            if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
            if occ(rr,cc), continue; end
            if (abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1)
                if occ(r,cc) || occ(rr,c), continue; end
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

    % --- String-pulling ---
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
        sp=pathXY(best_j,:); smoothPath(end+1,:)=sp; %#ok<AGROW>
        idx=best_j+1;
    end

    % --- Pure pursuit (v_plan) ---
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

    % --- ML nudge ---
    dist8=zeros(8,1);
    for kk=1:8
        dir=[cos(dirs(kk)), sin(dirs(kk))]; s=0; d=ray_max;
        while s<=ray_max
            q=pos+s*dir;
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

    %% --- FSM transitions ---
    near_wall_by_x = (pos(1)>=x_left - climb_band) && (pos(1)<=x_right + climb_band);
    below_top      = (pos(2) < climb_until_y);
    above_release  = (pos(2) > climb_until_y + release_hyst);

    switch state
        case STATE_NORMAL
            if near_wall_by_x && below_top
                x_pin = (pos(1) <= x_wall_center) * (x_left - hold_offset) + ...
                        (pos(1)  > x_wall_center) * (x_right + hold_offset);
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
    end

    %% --- DESCENT engage (with fallback if helper missing) ---
    if state==STATE_NORMAL
        above_goal = pos(2) > (goalXY(2) + 0.25);
        clear_of_slab= abs(pos(1) - x_wall_center) >= (wall_thickness/2 + safetyMargin*0.35);
        if above_goal && clear_of_slab
            useLoS = haveIsLineFree;
            okH=false; okV=false;
            if useLoS
                pH = [goalXY(1), pos(2)];
                pV = [goalXY(1), goalXY(2)];
                okH = isLineFree(pos, pH, isOcc, gridRes*0.25, min_clearance);
                okV = isLineFree(pH, pV, isOcc, gridRes*0.25, min_clearance);
            end
            if (~useLoS && abs(pos(1)-goalXY(1))<1.0) || (okH && okV)
                desc_x_lock   = goalXY(1);
                descent_until = t + DESC_MAX_T;
                state = STATE_DESCENT; prev_u=[0 0];
            end
        end
    end

    %% --- Command composition ---
    % planner + ML
    u_nom = v_plan + (USE_ML*w_ml)*v_ml;

    % downward intent: if we are above wall and goal is below, bias downward
    above_wall = pos(2) > lip_y_top + 0.05;
    goal_below = goalXY(2) < pos(2) - 0.1;
    if DOWNLOCK_ON && state==STATE_NORMAL && above_wall && goal_below
        x_correction = max(-1,min(1, (goalXY(1)-pos(1))/2.0 )); % gentle x toward goal
        down_vec = [0, -1];
        u_nom = DOWNLOCK_GAIN*(down_vec + [0.35*x_correction, 0]) + (1-DOWNLOCK_GAIN)*u_nom;
    end

    % dynamic obstacle side-pass latch
    dynBox = [min(dynPoly(:,1)) max(dynPoly(:,1)) min(dynPoly(:,2)) max(dynPoly(:,2))];
    expand = dyn_extra_margin + safetyMargin*0.6;
    x1 = dynBox(1)-expand; x2=dynBox(2)+expand; y1=dynBox(3)-expand; y2=dynBox(4)+expand;
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
    if pass_dir ~= 0 && state==STATE_NORMAL
        u_nom = 0.55*u_nom + 0.45*[pass_dir, 0];
    end

    % corner-pop (disable during commit states)
    rc = world2grid(pos);
    rc(1)=min(max(rc(1),1),mapRows); rc(2)=min(max(rc(2),1),mapCols);
    d_here = distMap(rc(1),rc(2));
    n_est = [gx(rc(1),rc(2)), gy(rc(1),rc(2))]; n_est = n_est / max(norm(n_est),1e-9);
    if state==STATE_NORMAL && d_here < POP_THRESH
        gain = POP_GAIN_MAX * (POP_THRESH - d_here) / POP_THRESH;
        u_nom = u_nom + gain * n_est;
    end

    % state-specific actuation
    if state==STATE_CLIMB
        u = [0, max(climb_speed, min_climb_vy)]; prev_u=u;

    elseif state==STATE_POSTREL
        to_goal = goalXY - pos; dg = max(norm(to_goal),1e-6);
        goal_dir = to_goal/dg;
        u = POSTREL_PUSH * [lat_sign, 0] + 0.30*goal_dir; prev_u=u;

    elseif state==STATE_LATERALCLR
        to_goal = goalXY - pos; dg = max(norm(to_goal),1e-6);
        goal_dir = to_goal/dg;
        u = 0.85 * [lat_sign, 0] + 0.30*goal_dir; prev_u=u;

    elseif state==STATE_DESCENT
        x_err = desc_x_lock - pos(1);
        u = [DESC_KX*max(-1,min(1,x_err)), -DESC_SPEED]; prev_u=u;

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

    % speed scaling (keep speed up during commit states)
    ttc_like=max(min(dmin/ray_max,1),0);
    base_floor = any(state==[STATE_POSTREL,STATE_LATERALCLR,STATE_DESCENT]) * 0.70 + ...
                 ~any(state==[STATE_POSTREL,STATE_LATERALCLR,STATE_DESCENT]) * 0.55;
    speed_scale = max(base_floor, base_floor + (1-base_floor)*ttc_like);
    spd=norm(u); if spd>vmax*speed_scale, u=(vmax*speed_scale/spd)*u; end
    if state==STATE_CLIMB, u(1)=0; u(2)=max(u(2),min_climb_vy); end

    % --- integrate (substeps) ---
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
                state = STATE_NORMAL; prev_u=[0 0]; break;

            else
                rc2=world2grid(pos);
                rc2(1)=min(max(rc2(1),1),mapRows); rc2(2)=min(max(rc2(2),1),mapCols);
                n2=[gx(rc2(1),rc2(2)), gy(rc2(1),rc2(2))];
                if norm(n2)<1e-6, n2=[-v_apply(2), v_apply(1)]; end
                n2=n2/max(norm(n2),1e-6); t2=[n2(2), -n2(1)];

                % sticky only in NORMAL
                if state==STATE_NORMAL
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
                    v_apply = 0.78*dot(v_apply,t2)*t2 + 0.14*v_plan + 0.08*n2;
                end

                tryStep = v_apply * (dt*(1 - (ss-1)/Nsub));
                seg_len2=norm(tryStep); Nsub2=max(1,ceil(seg_len2/max_step));
                subStep=tryStep/Nsub2; Nsub=ss-1+Nsub2; continue
            end
        end
        pos = p_try;
    end
    vel = v_apply;

    % log & goal check
    traj(k,:)=pos; dmin_hist(k)=dmin; speed_hist(k)=norm(vel); t_hist(k)=t;
    if norm(pos-goalXY)<=tolReach, break; end

    % draw
    if ~isgraphics(fig), fig=figure('Color','w','Visible','on'); end
    clf(fig); hold on; axis equal; xlim([xyMin(1) xyMax(1)]); ylim([xyMin(2) xyMax(2)]);
    for p=1:numel(staticPolys)
        P=staticPolys{p};
        patch(P(:,1),P(:,2),polyColors{p},'FaceAlpha',0.5,'EdgeColor','k');
    end
    if USE_DYN, patch(dynPoly(:,1),dynPoly(:,2),[.85 .4 .1],'FaceAlpha',0.6,'EdgeColor','k'); end
    % lip band (visual)
    patch([lip_x_min lip_x_max lip_x_max lip_x_min], [lip_y_top lip_y_top lip_y_top+lip_band_h lip_y_top+lip_band_h], ...
          [0.6 0.8 1.0], 'FaceAlpha',0.15, 'EdgeColor','none');
    plot(startXY(1),startXY(2),'go','MarkerFaceColor','g');
    plot(goalXY(1),goalXY(2),'rx','LineWidth',2,'MarkerSize',10);
    if exist('pathXY','var') && size(pathXY,1)>1, plot(pathXY(:,1), pathXY(:,2),'m:','LineWidth',1.1); end
    if exist('smoothPath','var') && size(smoothPath,1)>1, plot(smoothPath(:,1), smoothPath(:,2),'m-','LineWidth',1.8); end
    plot(traj(1:k,1),traj(1:k,2),'r-','LineWidth',2);
    if state==STATE_CLIMB,  xline(x_pin,'--k'); end
    quiver(pos(1),pos(2),u(1),u(2),0,'k');
    legend({'blue wall','gray box','dyn','lip band','start','goal','A* path','smoothed','traj','u'},'Location','northwest');
    title(sprintf('t=%.1fs  dmin=%.2f  speed=%.2f  state=%d', t, dmin, norm(vel), state));
    drawnow;
    try, frame=getframe(gcf); catch, drawnow; pause(0.01); frame=getframe(gcf); end
    writeVideo(vOut, frame);
end

close(vOut); saveas(fig,'week9_final_path.png');

% CSV
if LOG_CSV
    valid = ~isnan(t_hist);
    T = table(t_hist(valid), traj(valid,1), traj(valid,2), speed_hist(valid), dmin_hist(valid), ...
        'VariableNames', {'t','x','y','speed','dmin'});
    writetable(T,'week9_metrics.csv');
end
