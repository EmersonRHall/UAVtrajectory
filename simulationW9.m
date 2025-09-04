%% Week 9 — Integration with Simulation (tight dodging via A* + string-pulling + pure pursuit)
% Outputs: week9_demo.mp4, week9_final_path.png, week9_metrics.csv

clear; clc; close all; clear fig;

%% --- Load Week 6 dataset and rebuild ridge model (deploy trained policy) ---
S = load('week6_dataset.mat');                % expects features/actions/split
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

%% --- World + obstacles ---
gridRes=0.20;                         % grid cell (m) — coarser = faster planning
safetyMargin=0.30;                    % inflation for planning — smaller = tighter dodging
xyMin=[-10 -10]; xyMax=[10 10];
startXY=[-8,-8]; goalXY=[10,4];

% Tall vertical wall (blue) and a building (gray)
wall=[0.5 -6;0.7 -6;0.7 6;0.5 6];
bldg=[4 4;7 4;7 7;4 7];

inflate=@(poly,r)[min(poly(:,1))-r,min(poly(:,2))-r;
                  max(poly(:,1))+r,min(poly(:,2))-r;
                  max(poly(:,1))+r,max(poly(:,2))+r;
                  min(poly(:,1))-r,max(poly(:,2))+r];
wallInf=inflate(wall,safetyMargin); bldgInf=inflate(bldg,safetyMargin);

xv=xyMin(1):gridRes:xyMax(1); yv=xyMin(2):gridRes:xyMax(2);
[XC,YC]=meshgrid(xv,yv); [mapRows,mapCols]=size(XC);

% grid/world converters
world2grid=@(xy)[ round((xy(2)-xyMin(2))/gridRes)+1, round((xy(1)-xyMin(1))/gridRes)+1 ];
grid2world=@(rc)[ xyMin(1)+(rc(:,2)-1)*gridRes, xyMin(2)+(rc(:,1)-1)*gridRes ];

%% --- Dynamic obstacle (moved down near y ≈ -5) + wind ---
% Wider, lower box that gates the bottom passage; oscillates horizontally
dynBase=[-1.8 -6.2; 1.8 -6.2; 1.8 -3.8; -1.8 -3.8];  % centered around y≈-5
dynAmp = 2.0;                   % horizontal sweep (meters)
dynPer = 8.0;                   % seconds per full cycle

% Wind disturbance (applied to vehicle dynamics)
wind_xy=[0.20 -0.10]; wind_sigma=0.04;  % mean drift + gust

%% --- Control / planner params ---
dt=0.15; Tmax=50; Nsteps=round(Tmax/dt);
vmax=0.90; tolReach=0.30;

% Pure pursuit & smoothing
lookahead_m = 1.2;                   % how far along path to chase
max_turn = deg2rad(30);              % cap heading change per step
lowpass = 0.25;                      % velocity low-pass (0..1)

% Learned model blend (keep small so planner dominates)
w_ml = 0.18;

% Rays for ML features / speed scaling
dirs = deg2rad(0:45:315); ray_max=6; ray_step=gridRes*0.5;

% String-pulling / corridor settings
shortcut_step = 0.12;                % step when checking straight-line segment (m)
min_clearance = 0.02;                % tiny extra margin inside free space (m)

% A* setup (8-connected, no diagonal corner cut; small clearance bias)
neighbors = [ -1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1 ];
stepCost  = [ sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2) ];

%% --- Experiment toggles & logging ---
USE_WIND = true;    % apply wind to vehicle
USE_DYN  = true;    % moving orange obstacle
USE_ML   = true;    % blend learned velocity
LOG_CSV  = true;

pos=startXY; vel=[0 0]; prev_u=[0 0];
traj=zeros(Nsteps,2); dmin_hist=nan(Nsteps,1); speed_hist=nan(Nsteps,1); t_hist=nan(Nsteps,1);

%% --- Video ---
fig=figure('Color','w','Visible','on'); try, set(fig,'Position',[100 100 860 860]); end
vOut=VideoWriter('week9_demo.mp4','MPEG-4'); vOut.FrameRate=round(1/dt); open(vOut);

for k=1:Nsteps
    t=(k-1)*dt;

    % --- Dynamic obstacle state at time t (moved down; horizontal oscillation) ---
    dx=dynAmp*sin(2*pi*t/dynPer);
    dynPoly=dynBase+[dx 0;dx 0;dx 0;dx 0];
    dynInf=inflate(dynPoly, safetyMargin+0.08);  % slight extra inflate for planning

    % --- Occupancy grid (logical) ---
    wallMask = inpolygon(XC,YC,wallInf(:,1),wallInf(:,2));
    bldgMask = inpolygon(XC,YC,bldgInf(:,1),bldgInf(:,2));
    dynMask  = inpolygon(XC,YC,dynInf(:,1),dynInf(:,2));
    occ = wallMask | bldgMask | (USE_DYN & dynMask);   % elementwise toggle

    % Helper: point occupancy with bounds check
    isOcc = @(p) ( ...
        p(1) < xyMin(1) || p(1) > xyMax(1) || ...
        p(2) < xyMin(2) || p(2) > xyMax(2) || ...
        occ( ...
            min(max(round((p(2)-xyMin(2))/gridRes)+1,1), mapRows), ...
            min(max(round((p(1)-xyMin(1))/gridRes)+1,1), mapCols) ...
        ) ...
    );

    % --- Minimal range scan (get dmin for speed scaling; also for ML feats later) ---
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

    % --- A* planning from current pos to goal (every step) ---
    sRC = world2grid(pos); gRC = world2grid(goalXY);
    sRC(1)=min(max(sRC(1),1),mapRows); sRC(2)=min(max(sRC(2),1),mapCols);
    gRC(1)=min(max(gRC(1),1),mapRows); gRC(2)=min(max(gRC(2),1),mapCols);

    % if start cell is occupied (e.g., hugging obstacle), nudge to nearest free cell
    if occ(sRC(1),sRC(2))
        found=false;
        for rad=1:3
            rr=max(1,sRC(1)-rad):min(mapRows,sRC(1)+rad);
            cc=max(1,sRC(2)-rad):min(mapCols,sRC(2)+rad);
            [RR,CC]=ndgrid(rr,cc); ring=[RR(:),CC(:)];
            free = ~occ(sub2ind(size(occ), ring(:,1), ring(:,2)));
            if any(free), sRC = ring(find(free,1,'first'),:); found=true; break; end
        end
        if ~found, sRC=sRC; end
    end

    gScore = inf(mapRows,mapCols); fScore = inf(mapRows,mapCols);
    cameFrom = zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
    gScore(sRC(1),sRC(2))=0;
    heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
    fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;

    it=0; foundPath=false;
    while it<9000
        it=it+1; ftmp=fScore; ftmp(~openSet)=inf;
        [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
        [r,c]=ind2sub(size(occ),idxLin);
        if r==gRC(1) && c==gRC(2)
            pathRC=[r c];
            while ~(r==sRC(1) && c==sRC(2))
                prev=double(squeeze(cameFrom(r,c,:))');
                if all(prev==0), break; end
                r=prev(1); c=prev(2); pathRC(end+1,:)=[r c]; %#ok<AGROW>
            end
            pathRC=flipud(pathRC); foundPath=true; break
        end
        openSet(r,c)=false;
        for nb=1:8
            rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
            if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
            if occ(rr,cc), continue; end
            % NO diagonal corner cutting
            if (abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1)
                if occ(r,cc) || occ(rr,c), continue; end
            end
            tentative=gScore(r,c)+stepCost(nb);
            if tentative<gScore(rr,cc)
                cameFrom(rr,cc,:)=int32([r c]);
                % slight clearance preference (small so it doesn’t flee obstacles)
                r0=max(rr-1,1); r1=min(rr+1,mapRows);
                c0=max(cc-1,1); c1=min(cc+1,mapCols);
                prox = double( occ(r0:r1, c0:c1) );
                clearance_pen = 0.02 * sum(prox(:));   % SMALL bias
                gScore(rr,cc)=tentative;
                fScore(rr,cc)=tentative + heur(rr,cc) + clearance_pen;
                openSet(rr,cc)=true;
            end
        end
    end

    % Convert to world path; if fail (very unlikely), go straight toward goal safely
    if foundPath
        pathXY = grid2world(pathRC);
    else
        pathXY = [pos; goalXY];
    end

    % --- STRING-PULLING (shortcut/line-of-sight) to make tight dodges ---
    % Greedy: from current pos, jump to the farthest path point with a clear straight segment
    sp = pos; smoothPath = sp; idx=2;
    while idx <= size(pathXY,1)
        % try to push 'j' as far as possible
        j = idx;
        best_j = idx;
        while j <= size(pathXY,1)
            pA = sp; pB = pathXY(j,:);
            % collision check on segment pA->pB
            seg_free = true;
            seg_len = norm(pB - pA);
            if seg_len > 0
                nsteps = max(2, ceil(seg_len / shortcut_step));
                for ss=1:nsteps
                    q = pA + (ss/nsteps)*(pB - pA);
                    % expand check by tiny margin: sample a small cross around q
                    if isOcc(q) || isOcc(q + [min_clearance 0]) || isOcc(q + [-min_clearance 0]) || ...
                                   isOcc(q + [0 min_clearance]) || isOcc(q + [0 -min_clearance])
                        seg_free = false; break;
                    end
                end
            end
            if seg_free
                best_j = j; j = j + 1;   % try farther
            else
                break;
            end
        end
        sp = pathXY(best_j,:); smoothPath(end+1,:) = sp; %#ok<AGROW>
        idx = best_j + 1;
    end

    % --- Pure pursuit on smoothed path: pick lookahead target ---
    v_plan=[0 0];
    if size(smoothPath,1)>=2
        acc=0; ii=2; target_wp = smoothPath(min(2,size(smoothPath,1)),:);
        while ii<=size(smoothPath,1)
            seg = norm(smoothPath(ii,:)-smoothPath(ii-1,:));
            if acc + seg >= lookahead_m
                % interpolate within this segment
                need = lookahead_m - acc;
                dirseg = (smoothPath(ii,:)-smoothPath(ii-1,:))/max(seg,1e-6);
                target_wp = smoothPath(ii-1,:) + need*dirseg;
                break;
            else
                acc=acc+seg; target_wp=smoothPath(ii,:); ii=ii+1;
            end
        end
        dirp = target_wp - pos; ndp = norm(dirp);
        if ndp>1e-6, v_plan = dirp/ndp; end
    else
        dirp = goalXY - pos; ndp = norm(dirp);
        if ndp>1e-6, v_plan = dirp/ndp; end
    end

    % --- Learned model (deployed) as a small nudge ---
    % Build 8-ray features for ML
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
    yhat=[fx,1]*W_best; v_ml = yhat(1:2);

    % --- Compose command: path dominates; ML nudges only ---
    u_nom = v_plan + (USE_ML * w_ml) * v_ml;

    % Turn-rate limiter
    if norm(u_nom)>1e-6 && norm(prev_u)>1e-6
        hu=u_nom/norm(u_nom); hp=prev_u/norm(prev_u);
        ang = atan2(hu(2),hu(1)) - atan2(hp(2),hp(1));
        ang = atan2(sin(ang),cos(ang));
        ang_clamped = max(min(ang, max_turn), -max_turn);
        if abs(ang_clamped) < abs(ang)
            speed = norm(u_nom);
            target_heading = atan2(hp(2),hp(1)) + ang_clamped;
            u_nom = speed*[cos(target_heading), sin(target_heading)];
        end
    end

    % Low-pass smoothing
    u = (1-lowpass)*u_nom + lowpass*prev_u; prev_u=u;

    % Speed scaling vs. obstacle proximity (don’t overreact sideways, just slow down)
    ttc_like = max(min(dmin/ray_max, 1), 0);    % 0..1
    speed_scale = 0.55 + 0.45*ttc_like;         % 0.55..1.0
    if norm(u)>vmax*speed_scale, u = (vmax*speed_scale/norm(u))*u; end

    % --- Integrate with WIND disturbance and collision safety ---
    wind_term = [0 0];
    if USE_WIND
        wind_term = wind_xy + wind_sigma*randn(1,2);
    end
    tryStep = (u + wind_term) * dt; alpha=1.0; stepped=false;
    for attempt=1:3
        p_try = pos + alpha*tryStep;
        if ~isOcc(p_try)
            pos = p_try; vel = alpha*(u + wind_term); stepped=true; break;
        else
            % short backoff & bias along corridor direction if available
            if norm(v_plan)>1e-6, u_bias = 0.85*v_plan + 0.15*u; else, u_bias=u; end
            tryStep = u_bias*dt;
            alpha = alpha*0.6;
        end
    end
    if ~stepped
        % final tiny safe nudge toward next waypoint
        dirTiny = (goalXY - pos); if norm(dirTiny)>1e-6, dirTiny = dirTiny/norm(dirTiny); end
        pos = pos + 0.08*dirTiny*dt; vel = 0.08*dirTiny;
    end

    % --- Log & check goal ---
    traj(k,:)=pos; dmin_hist(k)=dmin; speed_hist(k)=norm(vel); t_hist(k)=t;
    if norm(pos-goalXY)<=tolReach, break; end

    % --- Draw ---
    if ~isgraphics(fig), fig=figure('Color','w','Visible','on'); end
    clf(fig); hold on; axis equal; xlim([xyMin(1) xyMax(1)]); ylim([xyMin(2) xyMax(2)]);
    patch(wallInf(:,1),wallInf(:,2),'b','FaceAlpha',0.5,'EdgeColor','k');
    patch(bldgInf(:,1),bldgInf(:,2),[.5 .5 .5],'FaceAlpha',0.7,'EdgeColor','k');
    if USE_DYN, patch(dynInf(:,1),dynInf(:,2),[.85 .4 .1],'FaceAlpha',0.6,'EdgeColor','k'); end
    plot(startXY(1),startXY(2),'go','MarkerFaceColor','g');
    plot(goalXY(1),goalXY(2),'rx','LineWidth',2,'MarkerSize',10);

    % planned path (before smoothing) and smooth corridor
    if exist('pathXY','var') && size(pathXY,1)>1
        plot(pathXY(:,1), pathXY(:,2),'m:','LineWidth',1.1);
    end
    if exist('smoothPath','var') && size(smoothPath,1)>1
        plot(smoothPath(:,1), smoothPath(:,2),'m-','LineWidth',1.8);
    end

    plot(traj(1:k,1),traj(1:k,2),'r-','LineWidth',2);
    quiver(pos(1),pos(2),u(1),u(2),0,'k');

    legend({'wall','block','dyn','start','goal','A* path','smoothed','traj','u'},'Location','northwest');
    title(sprintf('t=%.1fs  dmin=%.2f  speed=%.2f',t, dmin, speed_hist(k)));
    drawnow;
    try
        frame = getframe(gcf);
    catch
        drawnow; pause(0.01); frame = getframe(gcf);
    end
    writeVideo(vOut, frame);
end

close(vOut);
saveas(fig,'week9_final_path.png');

% --- Metrics CSV for your report ---
if LOG_CSV
    valid = ~isnan(t_hist);
    T = table(t_hist(valid), traj(valid,1), traj(valid,2), speed_hist(valid), dmin_hist(valid), ...
        'VariableNames', {'t','x','y','speed','dmin'});
    writetable(T,'week9_metrics.csv');
end
