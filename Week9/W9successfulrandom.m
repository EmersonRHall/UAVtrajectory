%% Week 9 — Robust Path-Follow + Safe Indexing (Frenet, Repulsion, Legend, Clean Plotting)
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

%% Random static scene (solvable + safe start/goal)
r_pix = max(1, ceil(safetyMargin / gridRes)); se_static = strel('disk', r_pix, 0);
maxTries = 40; SG_MIN_CLR = safetyMargin + 0.45;
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
        P=staticPolys{p}; occ_raw=occ_raw|inpolygon(XC,YC,P(:,1),P(:,2));
    end
    occ_static=imdilate(occ_raw,se_static);

    clrMap=bwdist(occ_static)*gridRes;
    ok=false;
    for t=1:400
        startXY=[-8+16*rand,-8+16*rand]; goalXY=[-8+16*rand,-8+16*rand];
        if norm(goalXY-startXY)<7, continue; end
        sRC=world2grid(startXY); gRC=world2grid(goalXY);
        if any(sRC<1)|any(gRC<1)|sRC(1)>mapRows|gRC(1)>mapRows|sRC(2)>mapCols|gRC(2)>mapCols, continue; end
        if clrMap(sRC(1),sRC(2))>SG_MIN_CLR && clrMap(gRC(1),gRC(2))>SG_MIN_CLR, ok=true; break; end
    end
    if ok, break; end
end

%% Dynamic obstacle
dynW=3+2.5*rand; dynH=2.4+1.2*rand;
dynCx=-6+12*rand; dynCy=-7+10*rand;
dynBase=[dynCx-dynW/2 dynCy-dynH/2; dynCx+dynW/2 dynCy-dynH/2; dynCx+dynW/2 dynCy+dynH/2; dynCx-dynW/2 dynCy+dynH/2];
dynAmp=1.5+1.5*rand; dynPer=6+6*rand;

%% Params
dt=0.12; Tmax=120; Nsteps=round(Tmax/dt);
vmax=0.9; amax=1.5; tolReach=0.35;
lookahead_m=2.3; ASTAR_PERIOD=1.0; astar_t_next=0;
ALPHA_MAX=0.35; EMERGENCY_R=1.0; BRAKE_GAIN=0.9;
dirs=deg2rad(0:45:315); ray_max=6; ray_step=gridRes*0.5;
shortcut_step=0.06; min_clearance=0.25;

%% State + Video
pos=startXY; vel=[0 0]; prev_u=[1 0]; last_wp=[]; cursor_s=0; cursor_margin=0.3;
traj=zeros(Nsteps,2); speed_hist=nan(Nsteps,1);
fig=figure('Color','w','Visible','on'); try,set(fig,'Position',[100 70 860 860]);end
vOut=VideoWriter('week9_demo.mp4','MPEG-4'); vOut.FrameRate=round(1/dt); open(vOut);

%% Safe occupancy helper (linear indexing via sub2ind)
isOcc = @(p,OccMask) ( ...
    p(1) < xyMin(1) || p(1) > xyMax(1) || p(2) < xyMin(2) || p(2) > xyMax(2) || ...
    OccMask( sub2ind( size(OccMask), ...
        clamp(floor((p(2)-xyMin(2))/gridRes)+1,1,mapRows), ...
        clamp(floor((p(1)-xyMin(1))/gridRes)+1,1,mapCols) )) );

%% Main loop
for k=1:Nsteps
    t=(k-1)*dt;

    % Dynamic obstacle motion
    dx=dynAmp*sin(2*pi*t/dynPer);
    dynPoly=dynBase+[dx 0;dx 0;dx 0;dx 0];

    % Occupancy now
    occ_static_now=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p}; occ_static_now=occ_static_now|inpolygon(XC,YC,P(:,1),P(:,2));
    end
    occ_static_now=imdilate(occ_static_now,se_static);
    occ_dyn=inpolygon(XC,YC,dynPoly(:,1),dynPoly(:,2));
    occ = occ_static_now | occ_dyn;

    % THICK mask for safety corridor
    r_thick = max(1, ceil((min_clearance + 0.5*gridRes)/gridRes));
    se_thick = strel('disk', r_thick, 0);
    occ_thick = imdilate(occ, se_thick);

    % Distance map (smoothed) and gradients (THIN)
    distMap = imgaussfilt(bwdist(occ)*gridRes, 1.5);
    [gy,gx] = gradient(distMap,gridRes,gridRes);

    % De-penetrate if inside THICK
    if isOcc(pos, occ_thick)
        distAway = bwdist(~occ_thick)*gridRes;
        [gy0,gx0] = gradient(-distAway,gridRes,gridRes);
        for dep=1:10
            rc = rc_from_xy(pos);
            n = [gx0(rc(1),rc(2)) gy0(rc(1),rc(2))]; n = n/max(norm(n),1e-9);
            pos = pos + 0.35*gridRes*n;
            if ~isOcc(pos,occ_thick), break; end
        end
        vel=[0 0];
    end

    % Keep inside world (soft)
    margin = 0.5;
    if pos(1) < xyMin(1)+margin, vel(1)=abs(vel(1)); end
    if pos(1) > xyMax(1)-margin, vel(1)=-abs(vel(1)); end
    if pos(2) < xyMin(2)+margin, vel(2)=abs(vel(2)); end
    if pos(2) > xyMax(2)-margin, vel(2)=-abs(vel(2)); end
    pos(1)=clamp(pos(1),xyMin(1),xyMax(1)); pos(2)=clamp(pos(2),xyMin(2),xyMax(2));

    % Replan (throttled A*)
    if t>=astar_t_next || ~exist('pathXY','var')
        astar_t_next=t+ASTAR_PERIOD;
        sRC = rc_from_xy(pos); gRC = rc_from_xy(goalXY);
        gScore=inf(mapRows,mapCols); fScore=inf(mapRows,mapCols);
        cameFrom=zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
        gScore(sRC(1),sRC(2))=0;
        heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
        fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;
        found=false;
        for it=1:8000
            ftmp=fScore; ftmp(~openSet)=inf;
            [minVal,idxLin]=min(ftmp(:)); if isinf(minVal), break; end
            [r,c]=ind2sub(size(occ),idxLin);
            if r==gRC(1) && c==gRC(2)
                pathRC=[r c];
                while ~(r==sRC(1) && c==sRC(2))
                    prev=double(squeeze(cameFrom(r,c,:))'); if all(prev==0), break; end
                    r=prev(1); c=prev(2); pathRC(end+1,:)=[r c]; %#ok<AGROW>
                end
                pathRC=flipud(pathRC); found=true; break
            end
            openSet(r,c)=false;
            for nb=1:8
                rr=r+neighbors(nb,1); cc=c+neighbors(nb,2);
                if rr<1||rr>mapRows||cc<1||cc>mapCols, continue; end
                if occ(rr,cc), continue; end
                % no corner threading against THICK
                if abs(neighbors(nb,1))==1 && abs(neighbors(nb,2))==1
                    if occ_thick(r,cc) || occ_thick(rr,c), continue; end
                end
                tentative=gScore(r,c)+stepCost(nb);
                dclr=distMap(rr,cc); clear_softmin = safetyMargin + 0.10; clear_pref = 0.60;
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
        if found, pathXY=grid2world(pathRC); else, pathXY=[pos; goalXY]; end

        % Ensure initialized for plotting even on first frames
        if ~exist('pathXY','var'),     pathXY=[startXY; goalXY]; end

        % String pulling with safe occupancy
        sp=pos; smoothPath=sp;
        for i=2:size(pathXY,1)
            pA=sp; pB=pathXY(i,:); seg_len=norm(pB-pA);
            free=true;
            if seg_len>0
                nsteps=max(2,ceil(seg_len/0.12));
                for ss=1:nsteps
                    q=pA+(ss/nsteps)*(pB-pA);
                    if isOcc(q, occ_thick), free=false; break; end
                end
            end
            if ~free, break; end
            smoothPath(end+1,:)=pathXY(i,:); sp=pathXY(i,:);
        end
        if ~exist('smoothPath','var') || size(smoothPath,1)<2
            smoothPath=[startXY; goalXY];
        end
    end

    % Lookahead target from arc-length cursor
    seglen=sqrt(sum(diff(smoothPath).^2,2)); s_cum=[0; cumsum(seglen)];
    best_d=inf; cp_pt=smoothPath(1,:); cp_idx=1; cp_s=0;
    for i=2:size(smoothPath,1)
        A=smoothPath(i-1,:); B=smoothPath(i,:); AB=B-A; L=max(norm(AB),1e-9);
        tproj=clamp(dot(pos-A,AB)/max(dot(AB,AB),1e-9),0,1);
        Q=A+tproj*AB; d=norm(pos-Q);
        if d<best_d, best_d=d; cp_pt=Q; cp_idx=i; cp_s=s_cum(i-1)+tproj*L; end
    end
    cursor_s=max(cursor_s,cp_s-cursor_margin);
    s_target=cursor_s+lookahead_m;
    if s_target>=s_cum(end)
        target_wp=smoothPath(end,:);
    else
        j=find(s_cum>=s_target,1);
        lam=(s_target-s_cum(j-1))/max(s_cum(j)-s_cum(j-1),1e-9);
        target_wp=smoothPath(j-1,:)+lam*(smoothPath(j,:)-smoothPath(j-1,:));
    end
    % Waypoint hysteresis
    if isempty(last_wp), last_wp=target_wp; end
    if norm(target_wp-last_wp)<0.5, target_wp=0.8*last_wp+0.2*target_wp; end
    last_wp=target_wp;

    % Frenet path-follow + soft repulsion
    A=smoothPath(max(cp_idx-1,1),:); B=smoothPath(min(cp_idx,size(smoothPath,1)),:);
    t_hat=(B-A)/max(norm(B-A),1e-9); n_hat=[-t_hat(2), t_hat(1)];
    e_y=dot(pos-cp_pt,n_hat);
    K_lat=1.1; u_path=t_hat - K_lat*e_y*n_hat;
    v_wp=(target_wp-pos); v_wp=v_wp/max(norm(v_wp),1e-9);
    u_path=0.9*u_path+0.1*v_wp; u_path=u_path/max(norm(u_path),1e-9);

    rc = rc_from_xy(pos);
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
    if norm(u_nom)>1e-9 && norm(prev_u)>1e-9
        ang_u=atan2(u_nom(2),u_nom(1)); ang_p=atan2(prev_u(2),prev_u(1));
        dAng=atan2(sin(ang_u-ang_p),cos(ang_u-ang_p));
        ang_new=ang_p + sign(dAng)*min(abs(dAng), HEADING_RATE_LIMIT*dt);
        u_nom=[cos(ang_new) sin(ang_new)];
    end
    u_sm=(1-BETA)*prev_u + BETA*u_nom; u_sm=u_sm/max(norm(u_sm),1e-9); prev_u=u_sm;

    % Speed + emergency brake (THICK rays)
    dmin=ray_max;
    for kk=1:8
        dir=[cos(dirs(kk)), sin(dirs(kk))]; s=0;
        while s<=ray_max
            q=pos+s*dir; if isOcc(q,occ_thick), dmin=min(dmin,s); break; end
            s=s+ray_step;
        end
    end
    spd_des = vmax * (dmin>=EMERGENCY_R) + vmax * BRAKE_GAIN * max(dmin/EMERGENCY_R,0) * (dmin<EMERGENCY_R);
    spd_des = max(spd_des, 0.25*vmax);
    v_des = spd_des*u_sm; dv=v_des-vel; dv_max=amax*dt; nv=norm(dv); if nv>dv_max, dv=(dv_max/nv)*dv; end
    vel=vel+dv;

    % Integrate in substeps with slide using THICK mask
    tryStep=vel*dt; max_step=0.30*gridRes; Nsub=max(1,ceil(norm(tryStep)/max_step)); subStep=tryStep/Nsub;
    for ss=1:Nsub
        p_try=pos+subStep;
        if isOcc(p_try,occ_thick)
            rc2=rc_from_xy(pos);
            n2=[gx(rc2(1),rc2(2)), gy(rc2(1),rc2(2))]; if norm(n2)<1e-6, n2=[-vel(2), vel(1)]; end
            n2=n2/max(norm(n2),1e-6);
            tR2=[ n2(2), -n2(1)]; tL2=[-n2(2),  n2(1)];
            gdir=(target_wp-pos)/max(norm(target_wp-pos),1e-9);
            t2=tR2; if dot(tL2,gdir)>dot(tR2,gdir), t2=tL2; end
            v_slide = 0.88*dot(vel,t2)*t2 + 0.10*n2 + 0.02*gdir;
            pos = pos + v_slide * (dt*(1 - (ss-1)/Nsub));
        else
            pos=p_try;
        end
        pos(1)=clamp(pos(1),xyMin(1),xyMax(1));
        pos(2)=clamp(pos(2),xyMin(2),xyMax(2));
    end

    % Log and (optional) early exit on goal
    traj(k,:)=pos; speed_hist(k)=norm(vel);
    if norm(pos-goalXY)<=tolReach, break; end

    % --------- DRAW (handles + robust legend) ---------
    clf(fig); hold on; axis equal; xlim([xyMin(1) xyMax(1)]); ylim([xyMin(2) xyMax(2)]);
    for p=1:numel(staticPolys)
        P=staticPolys{p};
        patch(P(:,1),P(:,2),polyColors{p},'FaceAlpha',0.5,'EdgeColor','k');
    end
    patch(dynPoly(:,1),dynPoly(:,2),[.85 .4 .1],'FaceAlpha',0.6,'EdgeColor','k');

    % Paths & entities (capture handles)
    hA=[]; hS=[]; hT=[]; hSt=[]; hG=[]; hWp=[];
    if exist('pathXY','var') && size(pathXY,1)>1
        hA = plot(pathXY(:,1), pathXY(:,2), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
    end
    if exist('smoothPath','var') && size(smoothPath,1)>1
        hS = plot(smoothPath(:,1), smoothPath(:,2), '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
    end
    hT  = plot(traj(1:k,1), traj(1:k,2), 'r-', 'LineWidth', 2);
    hSt = plot(startXY(1), startXY(2), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 6);
    hG  = plot(goalXY(1),  goalXY(2),  'rx', 'LineWidth', 2, 'MarkerSize', 10);
    hWp = plot(target_wp(1), target_wp(2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 6);

    title(sprintf('t=%.1fs  |v|=%.2f  dmin=%.2f', t, norm(vel), dmin));

    % Build legend from existing handles (prevents mismatched labels)
    legH = []; legL = {};
    if ~isempty(hA),  legH(end+1) = hA;  legL{end+1} = 'A* path';        end
    if ~isempty(hS),  legH(end+1) = hS;  legL{end+1} = 'Smoothed path';  end
    legH(end+1) = hT;  legL{end+1} = 'UAV trajectory';
    legH(end+1) = hSt; legL{end+1} = 'Start';
    legH(end+1) = hG;  legL{end+1} = 'Goal';
    legH(end+1) = hWp; legL{end+1} = 'Target wp';
    legend(legH, legL, 'Location', 'southoutside', 'Orientation', 'horizontal', 'AutoUpdate', 'off');

    drawnow; frame=getframe(gcf); writeVideo(vOut, frame);
end

close(vOut); saveas(fig,'week9_final_path.png');
fprintf('Done. Steps: %d  Reached: %d\n', find(all(~isnan(traj),2),1,'last'), norm(pos-goalXY)<=tolReach);
