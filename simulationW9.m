%% Week 9 — Integration with Simulation (robust release + anti-loop)
% Hybrid: Global A* + string-pulling + pure-pursuit + ML nudge
% Robust climb FSM (NORMAL/CLIMB/RELEASE), symmetric corner-pop, softened sticky sliding
% Stagnation-escape to prevent looping; can move freely in ANY direction
% Outputs: week9_demo.mp4, week9_final_path.png, week9_metrics.csv

clear; clc; close all; clear fig; rng(2);

%% --- Load Week 6 dataset and rebuild ridge model (deploy trained policy) ---
S = load('week6_dataset.mat');                % expects features/actions/split (Week 6)
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

%% --- World + obstacles (polygons; gray is a box) ---
gridRes=0.20;                         % grid cell (m)
safetyMargin=0.60;                    % inflation via imdilate disk
xyMin=[-10 -10]; xyMax=[10 10];
startXY=[-8,-8]; goalXY=[10,4];

% Thicker Blue wall (vertical slab) — top at y=5
x_wall_center = 0.60;
wall_thickness = 0.40;
x_left  = x_wall_center - wall_thickness/2;
x_right = x_wall_center + wall_thickness/2;
blueWall = [x_left -10; x_right -10; x_right 5.0; x_left 5.0];

% Gray box (axis-aligned)
grayBox  = [4 4; 7 4; 7 7; 4 7];

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
dynAmp = 2.0;  dynPer = 8.0;     % horizontal sweep
wind_xy=[0.20 -0.10]; wind_sigma=0.04;

%% --- Control / planner params ---
dt=0.12; Tmax=50; Nsteps=round(Tmax/dt);
vmax=0.85; tolReach=0.30;

% Pure pursuit & smoothing
lookahead_m = 1.4;          % slightly longer to reduce jitter
max_turn = pi;              % allow full heading change
lowpass  = 0.18;            % a bit snappier than before

% ML blend
w_ml = 0.18;

% Rays
dirs = deg2rad(0:45:315); ray_max=6; ray_step=gridRes*0.5;

% Shortcut smoothing
shortcut_step = 0.10;
min_clearance = 0.12;

% A* (8-connected)
neighbors = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
stepCost  = [ sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2) ];

% Clearance-aware extra cost
clear_pref    = 0.55;
clear_softmin = safetyMargin + 0.10;

%% === CLIMB FSM parameters ===
STATE_NORMAL  = 0; 
STATE_CLIMB   = 1;
STATE_RELEASE = 2;
state         = STATE_NORMAL;

climb_band    = safetyMargin + 0.20;  % engage band around wall in x
climb_until_y = 6.60;                 % top lip (world y) before release
release_hyst  = safetyMargin + 0.30;  % must exceed this above top for release
reengage_gap  = 0.50;                 % must fall below (top - gap) to re-engage

hold_offset   = safetyMargin + 0.15;  % lateral standoff while climbing
climb_speed   = 0.85*vmax;
min_climb_vy  = 0.60*climb_speed;
x_pin         = NaN;                   % rail x during climb

%% === Corner-pop / sticky sliding ===
POP_THRESH    = safetyMargin + 0.25;
POP_GAIN_MAX  = 0.60;
STICK_T       = 0.9;     % shorter latch
STICK_THRESH  = 0.18;
stick_until=0; stick_sign=0;

%% === Stagnation escape ===
STAG_WIN  = round(3.0/dt);    % seconds window
STAG_MIN_DROP = 0.25;         % must improve by this much within window
ESC_T     = 1.4;              % seconds of escape override
ESC_ON    = false; ESC_UNTIL=0;

%% --- State ---
USE_WIND = true; USE_DYN = true; USE_ML = true; LOG_CSV = true;

pos=startXY; vel=[0 0]; prev_u=[0 0];
traj=zeros(Nsteps,2); dmin_hist=nan(Nsteps,1); speed_hist=nan(Nsteps,1); t_hist=nan(Nsteps,1);
dist_goal_hist = nan(Nsteps,1);

%% --- Video ---
fig=figure('Color','w','Visible','on'); try, set(fig,'Position',[100 100 860 860]); end
vOut=VideoWriter('week9_demo.mp4','MPEG-4'); vOut.FrameRate=round(1/dt); open(vOut);

% Inflate via disk structuring element
r_pix = max(1, ceil(safetyMargin / gridRes));
se = strel('disk', r_pix, 0);

for k=1:Nsteps
    t=(k-1)*dt;

    % Dynamic obstacle
    dx=dynAmp*sin(2*pi*t/dynPer);
    dynPoly=dynBase+[dx 0;dx 0;dx 0;dx 0];

    % Occupancy build + inflation
    occ0=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p};
        occ0 = occ0 | inpolygon(XC,YC,P(:,1),P(:,2));
    end
    if USE_DYN
        occ0 = occ0 | inpolygon(XC,YC,dynPoly(:,1),dynPoly(:,2));
    end
    occ = imdilate(occ0, se);

    % Distance map + gradients
    distMap = bwdist(occ)*gridRes;
    [gy,gx] = gradient(distMap, gridRes, gridRes);  % rows=y, cols=x

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

    % A* plan
    sRC=world2grid(pos); gRC=world2grid(goalXY);
    sRC(1)=min(max(sRC(1),1),mapRows); sRC(2)=min(max(sRC(2),1),mapCols);
    gRC(1)=min(max(gRC(1),1),mapRows); gRC(2)=min(max(gRC(2),1),mapCols);

    % If goal cell inflated, slide goal to nearest free in a small ring
    if occ(gRC(1),gRC(2))
        freeFound=false;
        for rad=1:4
            rr=max(1,gRC(1)-rad):min(mapRows,gRC(1)+rad);
            cc=max(1,gRC(2)-rad):min(mapCols,gRC(2)+rad);
            [RR,CC]=ndgrid(rr,cc); ring=[RR(:),CC(:)];
            free = ~occ(sub2ind(size(occ), ring(:,1), ring(:,2)));
            if any(free), gRC = ring(find(free,1,'first'),:); freeFound=true; break; end
        end
        if freeFound, goalXY = grid2world(gRC); end
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
    gScore(sRC(1),sRC(2))=0;
    heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
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
            % No diagonal corner cutting
            if (abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1)
                if occ(r,cc) || occ(rr,c), continue; end
            end
            tentative=gScore(r,c)+stepCost(nb);
            if tentative < gScore(rr,cc)
                dclr=distMap(rr,cc); extra=0;
                if dclr<clear_softmin
                    extra = clear_pref*(1/max(dclr,1e-3) - 1/clear_softmin);
                end
                cameFrom(rr,cc,:)=int32([r c]);
                gScore(rr,cc)=tentative;
                fScore(rr,cc)=tentative+heur(rr,cc)+extra;
                openSet(rr,cc)=true;
            end
        end
    end
    if foundPath, pathXY=grid2world(pathRC); else, pathXY=[pos; goalXY]; end

    % String-pulling
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

    % Pure pursuit direction
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

    % ML nudge features
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

    %% --- CLIMB FSM ---
    near_wall_by_x = (pos(1)>=x_left - climb_band) && (pos(1)<=x_right + climb_band);
    below_top      = (pos(2) < climb_until_y);
    above_release  = (pos(2) > climb_until_y + release_hyst);
    well_below     = (pos(2) < climb_until_y - reengage_gap);

    switch state
        case STATE_NORMAL
            if near_wall_by_x && below_top
                x_pin = (pos(1) <= x_wall_center) * (x_left - hold_offset) + ...
                        (pos(1)  > x_wall_center) * (x_right + hold_offset);
                state = STATE_CLIMB;
            end
        case STATE_CLIMB
            if above_release
                state = STATE_RELEASE;
            elseif ~near_wall_by_x && ~below_top
                state = STATE_NORMAL; % safety fallback
            end
        case STATE_RELEASE
            % Don't re-enter climb unless we truly drop below the lip again
            if near_wall_by_x && well_below
                state = STATE_CLIMB;
            elseif ~near_wall_by_x
                state = STATE_NORMAL;
            end
    end

    %% --- Command composition ---
    u_nom = v_plan + (USE_ML*w_ml)*v_ml;

    % Symmetric corner-pop (normal + release only)
    rc = world2grid(pos);
    rc(1)=min(max(rc(1),1),mapRows); rc(2)=min(max(rc(2),1),mapCols);
    d_here = distMap(rc(1),rc(2));
    n_est = [gx(rc(1),rc(2)), gy(rc(1),rc(2))]; n_est = n_est / max(norm(n_est),1e-9);
    if state~=STATE_CLIMB && d_here < POP_THRESH
        gain = POP_GAIN_MAX * (POP_THRESH - d_here) / POP_THRESH;
        u_nom = u_nom + gain * n_est;
    end

    % State-specific control
    if state==STATE_CLIMB
        u = [0, max(climb_speed, min_climb_vy)];   % pure UP
        prev_u = u;
    else
        % turn-rate limit + low-pass
        if norm(u_nom)>1e-6 && norm(prev_u)>1e-6
            hu=u_nom/norm(u_nom); hp=prev_u/norm(prev_u);
            ang=atan2(hu(2),hu(1))-atan2(hp(2),hp(1)); ang=atan2(sin(ang),cos(ang));
            ang_clamped=max(min(ang,max_turn),-max_turn);
            if abs(ang_clamped)<abs(ang)
                speed=norm(u_nom); target_heading=atan2(hp(2),hp(1))+ang_clamped;
                u_nom = speed*[cos(target_heading), sin(target_heading)];
            end
        end
        u = (1-lowpass)*u_nom + lowpass*prev_u;
        prev_u = u;
    end

    % Speed scaling
    ttc_like=max(min(dmin/ray_max,1),0);
    speed_scale = 0.48 + 0.52*ttc_like;
    spd=norm(u); if spd>vmax*speed_scale, u=(vmax*speed_scale/spd)*u; end

    if state==STATE_CLIMB
        u(1)=0; u(2)=max(u(2),min_climb_vy);
    end

    %% --- Integrate (substeps) with wind + sticky sliding + escape ---
    wind_term=[0 0];
    if USE_WIND
        wind_term = wind_xy + wind_sigma*randn(1,2);
        if state==STATE_CLIMB, wind_term(1)=0; end
    end
    v_apply = u + wind_term;
    if state==STATE_CLIMB && v_apply(2)<min_climb_vy, v_apply(2)=min_climb_vy; end
    if state==STATE_CLIMB, v_apply(1)=0; end

    % Stagnation detection
    if k>STAG_WIN
        prog = dist_goal_hist(k-1) - dist_goal_hist(k-STAG_WIN);
        if ~ESC_ON && (prog < STAG_MIN_DROP)
            ESC_ON = true; ESC_UNTIL = t + ESC_T;
        end
    end
    if ESC_ON && t >= ESC_UNTIL
        ESC_ON = false;
    end

    % If escaping, push more strongly toward goal + away from obstacle normal
    if ESC_ON && state~=STATE_CLIMB
        v_escape = 0.70*v_plan + 0.25*n_est + 0.05*[n_est(2), -n_est(1)]; % small sideways bias
        v_apply  = v_escape;  % override for a short burst
        stick_until = t;      % cancel sticky during escape
    end

    tryStep = v_apply*dt; seg_len=norm(tryStep);
    max_step=0.30*gridRes; Nsub=max(1,ceil(seg_len/max_step)); subStep=tryStep/Nsub;

    for ss=1:Nsub
        p_try = pos + subStep;

        % Only pin X while actually climbing and below the lip
        if state==STATE_CLIMB && pos(2) < climb_until_y
            p_try(1)=x_pin;
        end

        hit = isOcc(p_try) || isOcc(p_try+[min_clearance 0]) || ...
              isOcc(p_try+[-min_clearance 0]) || isOcc(p_try+[0 min_clearance]) || ...
              isOcc(p_try+[0 -min_clearance]);

        if hit
            if state==STATE_CLIMB
                p_try(1)=x_pin;
                p_try(2)=pos(2)+max(gridRes*0.25, subStep(2));
                if isOcc(p_try), p_try(2)=pos(2)+gridRes*0.25; end
            else
                rc2=world2grid(pos);
                rc2(1)=min(max(rc2(1),1),mapRows); rc2(2)=min(max(rc2(2),1),mapCols);
                n2=[gx(rc2(1),rc2(2)), gy(rc2(1),rc2(2))];
                if norm(n2)<1e-6, n2=[-v_apply(2), v_apply(1)]; end
                n2=n2/max(norm(n2),1e-6);
                t2=[n2(2), -n2(1)];

                % Sticky reduced and disabled during escape
                if ~ESC_ON
                    if t < stick_until
                        if sign(t2(1)) ~= stick_sign, t2 = -t2; end
                    else
                        if dot(t2, v_apply) < 0, t2 = -t2; end
                    end
                    if abs(t2(1)) > STICK_THRESH
                        stick_sign = sign(t2(1)); stick_until = t + STICK_T;
                    end
                end

                % Above the wall (RELEASE): bias planner more, sticky less
                if state==STATE_RELEASE
                    v_apply = 0.65*v_plan + 0.25*t2 + 0.10*n2;
                else
                    v_apply = 0.78*dot(v_apply,t2)*t2 + 0.14*v_plan + 0.08*n2;
                end

                tryStep = v_apply * (dt*(1 - (ss-1)/Nsub));
                seg_len2=norm(tryStep); Nsub2=max(1,ceil(seg_len2/max_step));
                subStep=tryStep/Nsub2; Nsub=ss-1+Nsub2;
                continue
            end
        end

        pos = p_try;  % accept substep
    end
    vel = v_apply;

    % Logs & goal check
    traj(k,:)=pos; dmin_hist(k)=dmin; speed_hist(k)=norm(vel); t_hist(k)=t;
    dist_goal_hist(k)=norm(goalXY-pos);
    if norm(pos-goalXY)<=tolReach, break; end

    % Draw
    if ~isgraphics(fig), fig=figure('Color','w','Visible','on'); end
    clf(fig); hold on; axis equal; xlim([xyMin(1) xyMax(1)]); ylim([xyMin(2) xyMax(2)]);
    for p=1:numel(staticPolys)
        P=staticPolys{p};
        patch(P(:,1),P(:,2),polyColors{p},'FaceAlpha',0.5,'EdgeColor','k');
    end
    if USE_DYN
        patch(dynPoly(:,1),dynPoly(:,2),[.85 .4 .1],'FaceAlpha',0.6,'EdgeColor','k');
    end
    plot(startXY(1),startXY(2),'go','MarkerFaceColor','g');
    plot(goalXY(1),goalXY(2),'rx','LineWidth',2,'MarkerSize',10);
    if exist('pathXY','var') && size(pathXY,1)>1, plot(pathXY(:,1), pathXY(:,2),'m:','LineWidth',1.1); end
    if exist('smoothPath','var') && size(smoothPath,1)>1, plot(smoothPath(:,1), smoothPath(:,2),'m-','LineWidth',1.8); end
    plot(traj(1:k,1),traj(1:k,2),'r-','LineWidth',2);
    if state==STATE_CLIMB,  xline(x_pin,'--k'); end
    quiver(pos(1),pos(2),u(1),u(2),0,'k');
    legend({'blue wall','gray box','dyn','start','goal','A* path','smoothed','traj','u'},'Location','northwest');
    title(sprintf('t=%.1fs  dmin=%.2f  speed=%.2f  state=%d  escape=%d', t, dmin, speed_hist(k), state, ESC_ON));
    drawnow;
    try, frame=getframe(gcf); catch, drawnow; pause(0.01); frame=getframe(gcf); end
    writeVideo(vOut, frame);
end

close(vOut);
saveas(fig,'week9_final_path.png');

% Metrics
if LOG_CSV
    valid = ~isnan(t_hist);
    T = table(t_hist(valid), traj(valid,1), traj(valid,2), speed_hist(valid), dmin_hist(valid), ...
        'VariableNames', {'t','x','y','speed','dmin'});
    writetable(T,'week9_metrics.csv');
end
