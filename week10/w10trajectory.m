%% Week 10 — 2×UAV, Week-9 Controller Hardened (No Tunneling, Safe Visibility, Goal-Hold)

clear; clc; close all; rng('shuffle');

%% World/grid
xyMin=[-10 -10]; xyMax=[10 10];
gridRes=0.20; safetyMargin=0.70;
xv=xyMin(1):gridRes:xyMax(1); yv=xyMin(2):gridRes:xyMax(2);
[XC,YC]=meshgrid(xv,yv); [mapRows,mapCols]=size(XC);
neighbors=[-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
stepCost =[sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2)];
world2grid=@(xy)[ floor((xy(2)-xyMin(2))/gridRes)+1, floor((xy(1)-xyMin(1))/gridRes)+1 ];
grid2world=@(rc)[ xyMin(1)+(rc(:,2)-0.5)*gridRes, xyMin(2)+(rc(:,1)-0.5)*gridRes ];

% Helpers
clamp = @(v,lo,hi) max(min(v,hi),lo);
rc_from_xy = @(xy) [ clamp(floor((xy(2)-xyMin(2))/gridRes)+1,1,mapRows), ...
                     clamp(floor((xy(1)-xyMin(1))/gridRes)+1,1,mapCols) ];

%% Scene (Week-9 style polys + safe starts/goals)
r_pix = max(1, ceil(safetyMargin / gridRes)); se_static = strel('disk', r_pix, 0);
maxTries = 60; SG_MIN_CLR = safetyMargin + 0.45;

NU = 3; minSGsep = 4.0;

for attempt=1:maxTries
    wall_thickness = 0.35 + 0.45*rand;
    x_c = -2 + 4*rand; top_y = -1 + 7*rand;
    x_left  = x_c - wall_thickness/2; x_right = x_c + wall_thickness/2;
    blueWall = [x_left xyMin(2); x_right xyMin(2); x_right top_y; x_left top_y];

    gx=-8+16*rand; gy=-8+16*rand; gW=1.8+2.7*rand; gH=1.8+2.7*rand;
    grayBox=[gx gy; gx+gW gy; gx+gW gy+gH; gx gy+gH];

    Nextra=2+randi(2); extras=cell(Nextra,1);
    for i=1:Nextra
        cx=-8+16*rand; cy=-8+16*rand; w=1.2+3.0*rand; h=1.2+3.0*rand;
        extras{i}=[cx cy; cx+w cy; cx+w cy+h; cx cy+h];
    end
    staticPolys=[{blueWall,grayBox},extras(:)'];
    polyColors=[{'b',[0.5 0.5 0.5]},repmat({[0.6 0.6 0.9]},1,Nextra)];

    occ_raw=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p}; occ_raw = occ_raw | inpolygon(XC,YC,P(:,1),P(:,2));
    end
    occ_static = imdilate(occ_raw,se_static);

    clrMap=bwdist(occ_static)*gridRes;
    starts = nan(NU,2); goals = nan(NU,2); ok=false;
    for t=1:2000
        candS=[-8+16*rand,-8+16*rand]; candG=[-8+16*rand,-8+16*rand];
        if norm(candS-candG)<8, continue; end
        sRC=world2grid(candS); gRC=world2grid(candG);
        if any(sRC<1)|any(gRC<1)|sRC(1)>mapRows|gRC(1)>mapRows|sRC(2)>mapCols|gRC(2)>mapCols, continue; end
        if clrMap(sRC(1),sRC(2))>SG_MIN_CLR && clrMap(gRC(1),gRC(2))>SG_MIN_CLR
            goodS=true; goodG=true;
            for j=1:NU
                if ~isnan(starts(j,1)), goodS = goodS && norm(candS-starts(j,:))>minSGsep; end
                if ~isnan(goals(j,1)),  goodG = goodG && norm(candG-goals(j,:))>minSGsep;  end
            end
            if goodS && goodG
                ii=find(isnan(starts(:,1)),1);
                starts(ii,:)=candS; goals(ii,:)=candG;
            end
            if all(~isnan(starts(:,1))) && all(~isnan(goals(:,1))), ok=true; break; end
        end
    end
    if ok, break; end
end
if ~ok, error('Could not find solvable scene. Rerun.'); end

%% Dynamic obstacle (Week-9 polygon + sinusoid)
dynW=3+2.5*rand; dynH=2.4+1.2*rand;
dynCx=-6+12*rand; dynCy=-7+10*rand;
dynBase=[dynCx-dynW/2 dynCy-dynH/2; dynCx+dynW/2 dynCy-dynH/2; dynCx+dynW/2 dynCy+dynH/2; dynCx-dynW/2 dynCy+dynH/2];
dynAmp=1.5+1.5*rand; dynPer=6+6*rand;

%% Params
dt=0.12; Tmax=160; Nsteps=round(Tmax/dt);
vmax=0.9; amax=1.5; tolReach=0.35;
lookahead_m=2.3; ASTAR_PERIOD=1.0;
ALPHA_MAX=0.35; EMERGENCY_R=1.0; BRAKE_GAIN=0.9;
dirs=deg2rad(0:45:315); ray_max=6;
min_clearance=0.25;

% Integration & ray sampling (hardened)
INT_MAX_STEP = 0.15*gridRes;       % <= 0.15*gridRes to prevent tunneling
RAY_STEP     = 0.25*gridRes;       % visibility/ray sample spacing

R_sep = 1.0;   % visual only

%% Wind (Week-9 minimal)
W_MEAN = 0.3 + 0.4*rand;  W_DIR  = 2*pi*rand;
W_BASE = W_MEAN*[cos(W_DIR) sin(W_DIR)];
GUST_TAU = 3.0; GUST_SIGMA = 0.6;
w_gust=[0 0];
wind_at = @(t) (W_BASE + w_gust);
SPD_AIR_MAX_FACTOR = 1.8;
MIN_GROUND_PROGRESS_FRAC = 0.25;

%% Goal-hold + stall
STALL_T=6.0; PROG_EPS=0.05; NUDGE_VEL=0.6;

%% State
pos=starts; vel=zeros(NU,2); prev_u=repmat([1 0],NU,1);
last_wp=nan(NU,2); cursor_s=zeros(NU,1); cursor_margin=0.3;
pathXY=cell(NU,1); smoothPath=cell(NU,1); astar_t_next=zeros(NU,1);
traj=nan(Nsteps,2,NU); speed_hist=nan(Nsteps,NU);
bestGoalDist = vecnorm((goals - pos).').'; stallClock=zeros(NU,1); reached=false(NU,1);

fig=figure('Color','w'); try,set(fig,'Position',[90 60 980 980]); end
vOut=VideoWriter('week10_multi_like_week9.mp4','MPEG-4'); vOut.FrameRate=round(1/dt); open(vOut);
colors=lines(max(NU,4));

%% Occupancy helpers
isOcc = @(p,OccMask) ( ...
    p(1) < xyMin(1) || p(1) > xyMax(1) || p(2) < xyMin(2) || p(2) > xyMax(2) || ...
    OccMask( sub2ind( size(OccMask), ...
        clamp(floor((p(2)-xyMin(2))/gridRes)+1,1,mapRows), ...
        clamp(floor((p(1)-xyMin(1))/gridRes)+1,1,mapCols) )) );

% segment visibility with sampling <= RAY_STEP
segFree = @(a,b,OccMask) ( ...
    (~isOcc(a,OccMask)) && (~isOcc(b,OccMask)) && ...
    all(arrayfun(@(s) ~isOcc(a + s*(b-a), OccMask), linspace(0,1,max(3,ceil(norm(b-a)/RAY_STEP))))) );

%% Main loop
for k=1:Nsteps
    t=(k-1)*dt;

    % Wind update
    w_gust = w_gust + (-w_gust/GUST_TAU)*dt + sqrt(2*GUST_SIGMA^2/GUST_TAU)*sqrt(dt)*randn(1,2);
    w_vec = wind_at(t);

    % Dynamic obstacle
    dx=dynAmp*sin(2*pi*t/dynPer);
    dynPoly = dynBase+[dx 0;dx 0;dx 0;dx 0];

    % THIN occupancy
    occ_static_now=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p}; occ_static_now = occ_static_now | inpolygon(XC,YC,P(:,1),P(:,2));
    end
    occ_dyn = inpolygon(XC,YC,dynPoly(:,1),dynPoly(:,2));
    occ = occ_static_now | occ_dyn;

    % THICK occupancy
    r_thick = max(1, ceil((min_clearance + 0.5*gridRes)/gridRes));
    se_thick = strel('disk', r_thick, 0);
    occ_thick = imdilate(occ, se_thick);

    % Distance map from THIN (for gradients/projection)
    distMap = imgaussfilt(bwdist(occ)*gridRes, 1.5);
    [gy,gx] = gradient(distMap,gridRes,gridRes);

    for i=1:NU
        % Goal hold
        if reached(i)
            pos(i,:)=goals(i,:); vel(i,:)=[0 0];
            traj(k,:,i)=pos(i,:); speed_hist(k,i)=0; continue;
        end

        % Hard depenetration if inside THICK
        if isOcc(pos(i,:),occ_thick)
            distAway = bwdist(~occ_thick)*gridRes;
            [gy0,gx0] = gradient(-distAway,gridRes,gridRes);
            for dep=1:12
                rc = rc_from_xy(pos(i,:));
                n = [gx0(rc(1),rc(2)) gy0(rc(1),rc(2))]; n=n/max(norm(n),1e-9);
                pos(i,:) = pos(i,:) + 0.4*gridRes*n;
                if ~isOcc(pos(i,:),occ_thick), break; end
            end
            vel(i,:)=[0 0];
        end

        % Keep in bounds (soft)
        margin=0.5;
        if pos(i,1) < xyMin(1)+margin, vel(i,1)=abs(vel(i,1)); end
        if pos(i,1) > xyMax(1)-margin, vel(i,1)=-abs(vel(i,1)); end
        if pos(i,2) < xyMin(2)+margin, vel(i,2)=abs(vel(i,2)); end
        if pos(i,2) > xyMax(2)-margin, vel(i,2)=-abs(vel(i,2)); end
        pos(i,1)=clamp(pos(i,1),xyMin(1),xyMax(1)); pos(i,2)=clamp(pos(i,2),xyMin(2),xyMax(2));

        % Other-UAV keep-out (planning only)
        R_sep_plan=1.0; uav_dilate = strel('disk', max(1,ceil(R_sep_plan/gridRes)), 0);
        occ_uav_i=false(mapRows,mapCols);
        for j=1:NU
            if j==i, continue; end
            rcU=rc_from_xy(pos(j,:));
            if any(rcU<1) || rcU(1)>mapRows || rcU(2)>mapCols, continue; end
            mask=false(mapRows,mapCols); mask(rcU(1),rcU(2))=true;
            occ_uav_i = occ_uav_i | imdilate(mask, uav_dilate);
        end
        occ_plan = occ_thick | occ_uav_i;

        % Replan (freeze + safe string-pull) when due
        if t>=astar_t_next(i) || isempty(pathXY{i})
            astar_t_next(i)=t+ASTAR_PERIOD;

            % freeze airspeed to avoid overshoot during route flip
            vel(i,:)=[0 0];

            sRC=rc_from_xy(pos(i,:)); gRC=rc_from_xy(goals(i,:));
            occ_plan(sRC(1),sRC(2))=false; occ_plan(gRC(1),gRC(2))=false;

            % A*
            gScore=inf(mapRows,mapCols); fScore=inf(mapRows,mapCols);
            cameFrom=zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
            gScore(sRC(1),sRC(2))=0; heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
            fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;
            found=false; pathRC=[];

            for it=1:12000
                ftmp=fScore; ftmp(~openSet)=inf;
                [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
                [r,c]=ind2sub(size(occ_plan),idxLin);
                if r==gRC(1) && c==gRC(2)
                    pathRC=[r c];
                    while ~(r==sRC(1) && c==sRC(2))
                        prev=double(squeeze(cameFrom(r,c,:))'); if all(prev==0), break; end
                        r=prev(1); c=prev(2); pathRC(end+1,:)=[r c];
                    end
                    pathRC=flipud(pathRC); found=true; break
                end
                openSet(r,c)=false;
                for nb=1:8
                    rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
                    if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
                    if occ_plan(rr,cc), continue; end
                    if abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1
                        if occ_plan(r,cc) || occ_plan(rr,c), continue; end
                    end
                    tentative=gScore(r,c)+stepCost(nb);
                    dclr = distMap(rr,cc); clear_softmin = safetyMargin + 0.10; clear_pref = 0.60;
                    if dclr<clear_softmin
                        tentative = tentative + clear_pref*(1/max(dclr,1e-3) - 1/clear_softmin);
                    end
                    if tentative < gScore(rr,cc)-1e-12
                        cameFrom(rr,cc,:)=int32([r c]);
                        gScore(rr,cc)=tentative; fScore(rr,cc)=tentative+heur(rr,cc);
                        openSet(rr,cc)=true;
                    end
                end
            end

            if found
                P=grid2world(pathRC);
            else
                % retry without other-UAV keep-out
                occ_retry = occ_thick; occ_retry(sRC(1),sRC(2))=false; occ_retry(gRC(1),gRC(2))=false;
                gScore(:)=inf; fScore(:)=inf; cameFrom(:)=0; openSet(:)=false;
                gScore(sRC(1),sRC(2))=0; fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;
                for it=1:12000
                    ftmp=fScore; ftmp(~openSet)=inf;
                    [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
                    [r,c]=ind2sub(size(occ_retry),idxLin);
                    if r==gRC(1) && c==gRC(2)
                        pathRC=[r c];
                        while ~(r==sRC(1) && c==sRC(2))
                            prev=double(squeeze(cameFrom(r,c,:))'); if all(prev==0), break; end
                            r=prev(1); c=prev(2); pathRC(end+1,:)=[r c];
                        end
                        pathRC=flipud(pathRC); found=true; break
                    end
                    openSet(r,c)=false;
                    for nb=1:8
                        rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
                        if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
                        if occ_retry(rr,cc), continue; end
                        if abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1
                            if occ_retry(r,cc) || occ_retry(rr,c), continue; end
                        end
                        tentative=gScore(r,c)+stepCost(nb);
                        dclr=distMap(rr,cc); clear_softmin=safetyMargin+0.10; clear_pref=0.60;
                        if dclr<clear_softmin
                            tentative = tentative + clear_pref*(1/max(dclr,1e-3) - 1/clear_softmin);
                        end
                        if tentative < gScore(rr,cc)-1e-12
                            cameFrom(rr,cc,:)=int32([r c]);
                            gScore(rr,cc)=tentative; fScore(rr,cc)=tentative+heur(rr,cc);
                            openSet(rr,cc)=true;
                        end
                    end
                end
                if found, P=grid2world(pathRC); else, P=[pos(i,:); goals(i,:)]; end
            end
            pathXY{i}=P;

            % String pulling with segFree against THICK
            sp=pos(i,:); sm=sp;
            for ii=2:size(P,1)
                if ~segFree(sp, P(ii,:), occ_thick), break; end
                sm(end+1,:)=P(ii,:); sp=P(ii,:);
            end
            if size(sm,1)<2, sm=[pos(i,:); goals(i,:)]; end
            smoothPath{i}=sm;
        end

        % Lookahead & target (Week-9) + target projection if inside THICK
        seglen=sqrt(sum(diff(smoothPath{i}).^2,2)); s_cum=[0; cumsum(seglen)];
        best_d=inf; cp_pt=smoothPath{i}(1,:); cp_idx=1; cp_s=0;
        for ii=2:size(smoothPath{i},1)
            A=smoothPath{i}(ii-1,:); B=smoothPath{i}(ii,:); AB=B-A;
            L=max(norm(AB),1e-9); tproj=clamp(dot(pos(i,:)-A,AB)/max(dot(AB,AB),1e-9),0,1);
            Q=A+tproj*AB; d=norm(pos(i,:)-Q);
            if d<best_d, best_d=d; cp_pt=Q; cp_idx=ii; cp_s=s_cum(ii-1)+tproj*L; end
        end
        cursor_s(i)=max(cursor_s(i),cp_s-cursor_margin);
        s_target=cursor_s(i)+lookahead_m;
        if s_target>=s_cum(end)
            target_wp=smoothPath{i}(end,:);
        else
            j=find(s_cum>=s_target,1);
            lam=(s_target-s_cum(j-1))/max(s_cum(j)-s_cum(j-1),1e-9);
            target_wp=smoothPath{i}(j-1,:)+lam*(smoothPath{i}(j,:)-smoothPath{i}(j-1,:));
        end
        % If target falls inside THICK, project outward using distMap gradient
        if isOcc(target_wp,occ_thick)
            for proj=1:10
                rc=rc_from_xy(target_wp);
                n=[gx(rc(1),rc(2)) gy(rc(1),rc(2))]; n=n/max(norm(n),1e-9);
                target_wp = target_wp + 0.5*gridRes*n;
                if ~isOcc(target_wp,occ_thick), break; end
            end
        end
        if any(isnan(last_wp(i,:))), last_wp(i,:)=target_wp; end
        if norm(target_wp-last_wp(i,:))<0.5, target_wp=0.8*last_wp(i,:)+0.2*target_wp; end
        last_wp(i,:)=target_wp;

        % Week-9 controller (Frenet + soft repulsion)
        A=smoothPath{i}(max(cp_idx-1,1),:); B=smoothPath{i}(min(cp_idx,size(smoothPath{i},1)),:);
        t_hat=(B-A)/max(norm(B-A),1e-9); n_hat=[-t_hat(2), t_hat(1)];
        e_y=dot(pos(i,:)-cp_pt,n_hat);
        K_lat=1.1; u_path=t_hat - K_lat*e_y*n_hat;
        v_wp=(target_wp-pos(i,:)); v_wp=v_wp/max(norm(v_wp),1e-9);
        u_path=0.9*u_path+0.1*v_wp; u_path=u_path/max(norm(u_path),1e-9);

        rc = rc_from_xy(pos(i,:));
        g_obs=[gx(rc(1),rc(2)), gy(rc(1),rc(2))]; g_obs=g_obs/max(norm(g_obs),1e-9);
        d_here=distMap(rc(1),rc(2)); d0=0.9; d1=0.3;
        if d_here>=d0, w_rep=0; elseif d_here<=d1, w_rep=1; else, s=(d0-d_here)/(d0-d1); w_rep=s*(2-s); end
        side_sign=sign(e_y); if side_sign==0, side_sign=1; end
        t_slide = (side_sign>=0)*[ n_hat(2), -n_hat(1) ] + (side_sign<0)*[ -n_hat(2), n_hat(1) ];
        t_slide=t_slide/max(norm(t_slide),1e-9);
        v_rep=0.6*g_obs + 0.6*t_slide; v_rep=v_rep/max(norm(v_rep),1e-9);
        alpha=min(ALPHA_MAX, w_rep*0.8);
        u_nom=(1-alpha)*u_path + alpha*v_rep; u_nom=u_nom/max(norm(u_nom),1e-9);

        % Heading filter (rate-limited EMA)
        HEADING_RATE_LIMIT=deg2rad(60); BETA=0.4;
        if norm(u_nom)>1e-9 && norm(prev_u(i,:))>1e-9
            ang_u=atan2(u_nom(2),u_nom(1)); ang_p=atan2(prev_u(i,2),prev_u(i,1));
            dAng=atan2(sin(ang_u-ang_p),cos(ang_u-ang_p));
            ang_new=ang_p + sign(dAng)*min(abs(dAng), HEADING_RATE_LIMIT*dt);
            u_nom=[cos(ang_new) sin(ang_new)];
        end
        u_sm=(1-BETA)*prev_u(i,:) + BETA*u_nom; u_sm=u_sm/max(norm(u_sm),1e-9); prev_u(i,:)=u_sm;

        % Speed + wind (Week-9) with goal taper
        % Braking rays use segFree() (THICK)
        dmin=ray_max;
        for kk2=1:8
            dir=[cos(dirs(kk2)), sin(dirs(kk2))];
            % march with RAY_STEP to first hit
            s=0; hit=ray_max;
            while s<=ray_max
                q=pos(i,:)+s*dir;
                if isOcc(q,occ_thick), hit=s; break; end
                s=s+RAY_STEP;
            end
            if hit<dmin, dmin=hit; end
        end
        spd_des = vmax * (dmin>=EMERGENCY_R) + vmax * BRAKE_GAIN * max(dmin/EMERGENCY_R,0) * (dmin<EMERGENCY_R);
        spd_des = max(spd_des, 0.25*vmax);

        distGoal = norm(pos(i,:)-goals(i,:));
        mgp = MIN_GROUND_PROGRESS_FRAC * vmax * clamp(distGoal/(3*tolReach),0,1);

        v_gnd_des = max(spd_des, mgp) * u_sm;
        v_air_cmd = v_gnd_des - w_vec;
        v_air_max = SPD_AIR_MAX_FACTOR * vmax;
        na=norm(v_air_cmd); if na>v_air_max, v_air_cmd=(v_air_max/na)*v_air_cmd; end
        gnd_prog = dot(v_air_cmd + w_vec, u_sm);
        if gnd_prog < mgp
            v_air_cmd = v_air_cmd + (mgp - gnd_prog)*u_sm;
            na=norm(v_air_cmd); if na>v_air_max, v_air_cmd=(v_air_max/na)*v_air_cmd; end
        end

        % Stall recovery (unchanged)
        if distGoal < bestGoalDist(i) - PROG_EPS
            bestGoalDist(i)=distGoal; stallClock(i)=0;
        else
            stallClock(i)=stallClock(i)+dt;
            if stallClock(i) > STALL_T
                astar_t_next(i)=t-1e-6;
                rc2=rc_from_xy(pos(i,:));
                n2=[gx(rc2(1),rc2(2)) gy(rc2(1),rc2(2))]; if norm(n2)<1e-6, n2=[-u_sm(2),u_sm(1)]; end
                n2=n2/max(norm(n2),1e-6);
                tR2=[ n2(2), -n2(1)]; tL2=[-n2(2),  n2(1)];
                gdir=(goals(i,:)-pos(i,:))/max(distGoal,1e-9);
                t2=tR2; if dot(tL2,gdir)>dot(tR2,gdir), t2=tL2; end
                v_air_cmd = v_air_cmd + NUDGE_VEL * t2;
                stallClock(i)=0;
            end
        end

        % Accel limit
        dv=v_air_cmd-vel(i,:); dv_max=amax*dt; nv=norm(dv); if nv>dv_max, dv=(dv_max/nv)*dv; end
        vel(i,:)=vel(i,:)+dv;

        % Integrate ground motion with anti-tunneling substeps + segment check
        tryStep=(vel(i,:)+w_vec)*dt;
        Nsub=max(1,ceil(norm(tryStep)/INT_MAX_STEP)); sub_dt=dt/Nsub; subStep=tryStep/Nsub;
        for ss=1:Nsub
            p_try = pos(i,:)+subStep;
            % segment check between pos and p_try (THICK)
            if ~segFree(pos(i,:), p_try, occ_thick)
                % slide along boundary using gradient
                rc2=rc_from_xy(pos(i,:));
                n2=[gx(rc2(1),rc2(2)) gy(rc2(1),rc2(2))]; if norm(n2)<1e-6, n2=[-vel(i,2), vel(i,1)]; end
                n2=n2/max(norm(n2),1e-6);
                tR2=[ n2(2), -n2(1)]; tL2=[-n2(2),  n2(1)];
                gdir=(target_wp-pos(i,:))/max(norm(target_wp-pos(i,:)),1e-9);
                t2=tR2; if dot(tL2,gdir)>dot(tR2,gdir), t2=tL2; end
                v_slide = 0.88*dot(vel(i,:),t2)*t2 + 0.10*n2 + 0.02*gdir;
                pos(i,:) = pos(i,:) + v_slide*sub_dt;  % use sub_dt (not a scaled dt)
            else
                pos(i,:)=p_try;
            end
            pos(i,1)=clamp(pos(i,1),xyMin(1),xyMax(1));
            pos(i,2)=clamp(pos(i,2),xyMin(2),xyMax(2));
        end

        % Log + goal capture
        traj(k,:,i)=pos(i,:); speed_hist(k,i)=norm(vel(i,:));
        if norm(pos(i,:)-goals(i,:))<=tolReach
            reached(i)=true; vel(i,:)=[0 0]; pos(i,:)=goals(i,:);
        end
    end

    if all(reached), break; end

    %% Draw
    clf(fig); hold on; axis equal; xlim([xyMin(1) xyMax(1)]); ylim([xyMin(2) xyMax(2)]);
    for p=1:numel(staticPolys)
        P=staticPolys{p}; patch(P(:,1),P(:,2),polyColors{p},'FaceAlpha',0.5,'EdgeColor','k');
    end
    patch(dynPoly(:,1),dynPoly(:,2),[.85 .4 .1],'FaceAlpha',0.6,'EdgeColor','k');

    legH=[]; legL={};
    for i=1:NU
        if ~isempty(pathXY{i}) && size(pathXY{i},1)>1
            hA=plot(pathXY{i}(:,1),pathXY{i}(:,2),'--','Color',[0.7 0.7 0.7],'LineWidth',1.3);
            legH(end+1)=hA; legL{end+1}=sprintf('A* U%d',i);
        end
        if ~isempty(smoothPath{i}) && size(smoothPath{i},1)>1
            hS=plot(smoothPath{i}(:,1),smoothPath{i}(:,2),'-','Color',[0.4 0.4 0.4],'LineWidth',1.8);
            legH(end+1)=hS; legL{end+1}=sprintf('Smooth U%d',i);
        end
        col=colors(i,:);
        hT=plot(traj(1:k,1,i),traj(1:k,2,i),'-','Color',col,'LineWidth',2.0);
        hSg=plot(starts(i,1),starts(i,2),'o','MarkerFaceColor',col,'Color',col,'MarkerSize',6);
        hG=plot(goals(i,1),goals(i,2),'x','Color',col,'LineWidth',1.6,'MarkerSize',10);
        legH=[legH hT hSg hG]; legL=[legL {sprintf('U%d traj',i) sprintf('U%d start',i) sprintf('U%d goal',i)}]; %#ok<AGROW>
        th=linspace(0,2*pi,40); circ=pos(i,:)+R_sep*[cos(th(:)) sin(th(:))];
        plot(circ(:,1),circ(:,2),':','Color',col,'LineWidth',1.0);
        if reached(i), plot(pos(i,1),pos(i,2),'s','MarkerSize',8,'MarkerFaceColor',col,'Color',col); end
    end

    title(sprintf('Week 10: time = %.1fs |wind|=%.2f', t, norm(w_vec)));
    legend(legH,legL,'Location','southoutside','Orientation','horizontal','AutoUpdate','off');

    % Wind inset
    axInset=axes('Position',[0.03 0.78 0.18 0.18],'Color','none','HitTest','off','PickableParts','none');
    hold(axInset,'on'); axis(axInset,'equal'); axis(axInset,[-1 1 -1 1]); set(axInset,'Visible','off');
    plot(axInset,[-0.9 0.9],[0 0],':','Color',[0.7 0.7 0.7],'LineWidth',0.8);
    plot(axInset,[0 0],[-0.9 0.9],':','Color',[0.7 0.7 0.7],'LineWidth',0.8);
    wmag=norm(w_vec); L=0.9*min(1,wmag/2.0); tip=(w_vec/max(wmag,1e-9))*L;
    quiver(axInset,0,0,tip(1),tip(2),0,'LineWidth',1.6,'MaxHeadSize',2);
    text(axInset,-0.98,-0.98,sprintf('%.2f m/s',wmag),'Units','normalized','HorizontalAlignment','left','VerticalAlignment','bottom','FontSize',8,'FontWeight','bold');

    drawnow; frame=getframe(gcf); writeVideo(vOut, frame);
end

close(vOut); saveas(fig,'week10_final.png');

% Done
lastStep = find(all(~isnan(traj(:,1,1)),2),1,'last');
fprintf('Done. Steps: %d\n', lastStep);
for i=1:NU
    fprintf('U%d REACHED: %d  (dist=%.2f)\n', i, reached(i), norm(pos(i,:)-goals(i,:)));
end

