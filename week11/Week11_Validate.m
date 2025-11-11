%% Week 11 — Validation (deterministic scenarios across weather/mode)
% - Same terrain + starts/goals for all weathers and both modes (fair A/B)
% - 720×720 stable video capture
% - All collision checks use THICK occupancy
% - Robust SG placement with BFS connectivity on inflated grid
% - Natural terrain fix: no features in pond; no overlapping "buildings" (rocks/logs)
% - Unique-path planning: reserve/dilate other UAVs' corridors to force distinct routes

clc; close all;

% ---------- terrain choice ----------
fprintf('Choose a terrain:\n 1) urban\n 2) natural\n 3) man_made\n');
txt = input('Enter 1-3 (or press Enter to run ALL): ','s');
if isempty(txt)
    terrains = {'urban','natural','man_made'};
else
    s = lower(strtrim(txt));
    switch s
        case {'1','urban'}, terrains = {'urban'};
        case {'2','natural'}, terrains = {'natural'};
        case {'3','man_made','man-made','manmade'}, terrains = {'man_made'};
        otherwise, error('Unknown choice: %s', s);
    end
end

% ---------- experiment setup ----------
weathers = {'calm','breeze','gusty','storm'};
modes    = {'baseline','machine learning'};

NU = 3;                      % UAV count
TRIALS_PER_TERRAIN = 1;      % scenarios per terrain
MAKE_VIDEOS = true;

figdir = 'week11_figs'; if ~exist(figdir,'dir'), mkdir(figdir); end
viddir = 'week11_videos'; if ~exist(viddir,'dir'), mkdir(viddir); end

opts = struct( ...
    'NU',           NU, ...
    'worldSize',    10, ...
    'gridRes',      0.20, ...
    'safetyMargin', 0.70, ...
    'figdir',       figdir, ...
    'viddir',       viddir, ...
    'makeVideo',    MAKE_VIDEOS);

% Total rows = terrain * trials * weather * modes
Rows = numel(terrains)*TRIALS_PER_TERRAIN*numel(weathers)*numel(modes);
T = table(strings(Rows,1),strings(Rows,1),strings(Rows,1),(1:Rows)', ...
    false(Rows,1), nan(Rows,1), nan(Rows,1), nan(Rows,1), nan(Rows,1), nan(Rows,1), ...
    'VariableNames',{'terrain','weather','mode','trial_id','success','time_to_goal','path_len','min_clear','mean_speed','replans'});

row = 0; trialCounter = 0;

% ======== Outer loop creates SCENARIOS only by (terrain, trial) ========
for it = 1:numel(terrains)
    terrName = terrains{it};
    for rep = 1:TRIALS_PER_TERRAIN
        baseSeed = str2seed(terrName, sprintf('trial%d',rep));
        S = generate_scenario(terrName, opts, baseSeed);

        for iw = 1:numel(weathers)
            for im = 1:numel(modes)
                trialCounter = trialCounter + 1;
                modeStr    = modes{im};
                weatherStr = weathers{iw};
                fprintf('[%3d] %s | %s | %s\n', trialCounter, terrName, weatherStr, modeStr);

                runSeed = str2seed(terrName, sprintf('trial%d',rep), weatherStr, modeStr);
                R = simulate_scenario(opts, S, weatherStr, modeStr, runSeed);

                row = row + 1;
                T.terrain(row)     = terrName;
                T.weather(row)     = weatherStr;
                T.mode(row)        = modeStr;
                T.trial_id(row)    = row;
                T.success(row)     = all(R.reached);
                T.time_to_goal(row)= R.time_to_goal;
                T.path_len(row)    = R.total_path_len;
                T.min_clear(row)   = R.min_clear;
                T.mean_speed(row)  = R.mean_speed;
                T.replans(row)     = R.replan_count;
            end
        end
    end
end

writetable(T,'week11_metrics.csv'); disp('Saved week11_metrics.csv');
plot_success_bars(T, figdir);
plot_box_time_len(T, figdir);
plot_clearance_cdf(T, figdir);
disp('Week 11 validation complete.');

% ======================================================================
function S = generate_scenario(terrainName, opts, baseSeed)
rng(baseSeed,'twister');

xyMin=[-opts.worldSize -opts.worldSize]; xyMax=[opts.worldSize opts.worldSize];
gridRes=opts.gridRes; safetyMargin=opts.safetyMargin;

xv=xyMin(1):gridRes:xyMax(1); yv=xyMin(2):gridRes:xyMax(2);
[XC,YC]=meshgrid(xv,yv); [mapRows,mapCols]=size(XC);

% Build terrain deterministic (with pond-aware natural)
[staticPolys, polyColors, pondInfo] = build_terrain(terrainName, xyMin, xyMax);

% Inflate THIN for SG safety
r_pix = max(1, ceil(safetyMargin / gridRes));
se_static = strel('disk', r_pix, 0);

occ_raw=false(mapRows,mapCols);
for p=1:numel(staticPolys)
    P=staticPolys{p}; occ_raw = occ_raw | inpolygon(XC,YC,P(:,1),P(:,2));
end
occ_static = imdilate(occ_raw,se_static);

% Place SG deterministically (adaptive ladder + BFS on inflated grid)
NU = opts.NU;
if strcmpi(terrainName,'natural')
    minSGsep_base = 7.0;  % wider spacing to avoid overlap
else
    minSGsep_base = 4.5;
end
SG_MIN_CLR_base = safetyMargin + 0.50;

relax_SG_CLR = [1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50];
relax_SG_SEP = [1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50];

cornerBoxes  = default_corner_boxes(xyMin, xyMax, 2.0);
if strcmpi(terrainName,'natural') && ~isempty(pondInfo)
    cornerBoxes = corner_boxes_away_from_pond(cornerBoxes, pondInfo, 0.65);
end

placed=false;
for pass = 1:numel(relax_SG_CLR)
    SG_MIN_CLR = SG_MIN_CLR_base * relax_SG_CLR(pass);
    minSGsep   = minSGsep_base   * relax_SG_SEP(pass);
    [starts, goals, ok] = place_sg_random_safely_adaptive( ...
        NU, xyMin, xyMax, mapRows, mapCols, gridRes, staticPolys, XC, YC, ...
        SG_MIN_CLR, minSGsep, se_static, occ_static, cornerBoxes);
    if ok, placed=true; break; end
end
if ~placed
    error('Week11_Validate:Scenario', 'Failed to place starts/goals deterministically for scenario.');
end

% --- Inject an obstacle between the YELLOW UAV's start and goal (natural only) ---
if strcmpi(terrainName,'natural') && opts.NU >= 3 && ~isempty(pondInfo)
    yellowIdx = 3;                              % lines() default: 3rd color is yellow
    aYG = starts(yellowIdx,:); bYG = goals(yellowIdx,:);
    if all(isfinite(aYG)) && all(isfinite(bYG))
        [staticPolys, polyColors] = add_block_between_points( ...
            staticPolys, polyColors, pondInfo, aYG, bYG, ...
            'longLen', 3.0, ...   % along the blocking direction (perpendicular to path)
            'shortLen', 0.60, ... % thickness of the obstacle
            'clearStart', 1.0, ... % keep at least 1.0 m away from start
            'clearGoal', 1.0);    % keep at least 1.0 m away from goal
    end
end

S.xyMin = xyMin; S.xyMax = xyMax;
S.gridRes = gridRes;
S.staticPolys = staticPolys; S.polyColors = polyColors;
S.starts = starts; S.goals = goals;
S.XC = XC; S.YC = YC; S.mapRows = mapRows; S.mapCols = mapCols;
S.pondInfo = pondInfo;
end

function boxes = default_corner_boxes(xyMin, xyMax, edgePad)
boxes  = [ ...
    xyMin(1)+edgePad,               xyMin(2)+edgePad,               3.5, 3.5;   % bottom-left
    xyMax(1)-edgePad-3.5,           xyMin(2)+edgePad,               3.5, 3.5;   % bottom-right
    xyMin(1)+edgePad,               xyMax(2)-edgePad-3.5,           3.5, 3.5;   % top-left
    xyMax(1)-edgePad-3.5,           xyMax(2)-edgePad-3.5,           3.5, 3.5];  % top-right
end

function boxes2 = corner_boxes_away_from_pond(boxes, pondInfo, shrinkFrac)
cx=pondInfo.cx; cy=pondInfo.cy; rx=pondInfo.rx; ry=pondInfo.ry;
boxes2=boxes;
for k=1:size(boxes,1)
    x=boxes(k,1); y=boxes(k,2); w=boxes(k,3); h=boxes(k,4);
    px=x+w/2; py=y+h/2;
    dir=[px-cx, py-cy]; n=norm(dir); if n<1e-9, dir=[1 0]; n=1; end
    dir=dir/n;
    if hypot(px-cx, py-cy) < 1.3*max(rx,ry)+0.7
        shift=1.0*max(rx,ry);
        px2=px+shift*dir(1); py2=py+shift*dir(2);
    else
        px2=px; py2=py;
    end
    w2=w*shrinkFrac; h2=h*shrinkFrac;
    boxes2(k,:)=[px2-w2/2, py2-h2/2, w2, h2];
end
end

% ======================================================================
function R = simulate_scenario(opts, S, weatherStr, modeStr, runSeed)
rng(runSeed,'twister');

% ---------- World/grid ----------
xyMin=S.xyMin; xyMax=S.xyMax; gridRes=S.gridRes;
XC=S.XC; YC=S.YC; mapRows=S.mapRows; mapCols=S.mapCols;
staticPolys=S.staticPolys; polyColors=S.polyColors;
starts=S.starts; goals=S.goals;

neighbors=[-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
stepCost =[sqrt(2); 1; sqrt(2); 1; 1; sqrt(2); 1; sqrt(2)];

clamp = @(v,lo,hi) max(min(v,hi),lo);
world2grid=@(xy)[ floor((xy(2)-xyMin(2))/gridRes)+1, floor((xy(1)-xyMin(1))/gridRes)+1 ];
grid2world=@(rc)[ xyMin(1)+(rc(:,2)-0.5)*gridRes, xyMin(2)+(rc(:,1)-0.5)*gridRes ];
rc_from_xy = @(xy) [ clamp(floor((xy(2)-xyMin(2))/gridRes)+1,1,mapRows), ...
                     clamp(floor((xy(1)-xyMin(1))/gridRes)+1,1,mapCols) ];

NU = opts.NU; safetyMargin=opts.safetyMargin;

% ---------- Sim params ----------
dt=0.12;
Tmax=160;
Nsteps=round(Tmax/dt);
vmax=0.9;
amax = iff(any(strcmpi(weatherStr,{'gusty','storm'})), 2.2, 1.6);
tolReach=0.35;

lookahead_m = iff(strcmpi(modeStr,'ml'), 2.6, 2.3);
ASTAR_PERIOD=0.9;

ALPHA_MAX=0.35; EMERGENCY_R=1.0; BRAKE_GAIN=0.9;
dirs = deg2rad(0:45:315);
ray_max=6;
min_clearance=0.25;

INT_MAX_STEP = 0.08*gridRes;  % anti-tunneling
RAY_STEP     = 0.18*gridRes;  % visibility sampling

% ---------- Wind ----------
WP = wind_profile(weatherStr);
W_BASE = WP.W_BASE; GUST_SIGMA=WP.GUST_SIGMA; GUST_TAU=WP.GUST_TAU;
w_gust = [0 0];
wind_at = @(t) (W_BASE + w_gust);
SPD_AIR_MAX_FACTOR = WP.SPD_AIR_MAX_FACTOR;
MIN_GROUND_PROGRESS_FRAC = WP.MIN_GROUND_PROGRESS_FRAC;
WIND_THRUST_BOOST = WP.WIND_THRUST_BOOST;
WIND_WEIGHT_ASTAR = WP.WIND_WEIGHT_ASTAR;

Kwind_I_base = WP.Kwind_I;           % wind integral memory (per-run)
windInt = zeros(NU,2);

% ---------- Stall / goal-hold ----------
STALL_T=6.0; PROG_EPS=0.05; NUDGE_VEL=0.6;

% ---------- State ----------
pos=starts; vel=zeros(NU,2); prev_u=repmat([1 0],NU,1);
last_wp=nan(NU,2); cursor_s=zeros(NU,1); cursor_margin=0.3;
pathXY=cell(NU,1); smoothPath=cell(NU,1); astar_t_next=zeros(NU,1);
traj=nan(Nsteps,2,NU); speed_hist=nan(Nsteps,NU);
bestGoalDist = vecnorm((goals - pos).').'; stallClock=zeros(NU,1); reached=false(NU,1);
replans=zeros(NU,1); path_len=zeros(NU,1); min_clear_all=inf(NU,1);
lowProgClock = zeros(NU,1);

% ---------- Helpers ----------
isOcc = @(p,OccMask) ( ...
    p(1) < xyMin(1) || p(1) > xyMax(1) || p(2) < xyMin(2) || p(2) > xyMax(2) || ...
    OccMask( sub2ind( size(OccMask), ...
        clamp(floor((p(2)-xyMin(2))/gridRes)+1,1,mapRows), ...
        clamp(floor((p(1)-xyMin(1))/gridRes)+1,1,mapCols) )) );

segFree = @(a,b,OccMask) ( ...
    (~isOcc(a,OccMask)) && (~isOcc(b,OccMask)) && ...
    all(arrayfun(@(s) ~isOcc(a + s*(b-a), OccMask), linspace(0,1,max(3,ceil(norm(b-a)/RAY_STEP))))) );

% ---------- Video ----------
TARGET_HW = [720 720];
vidfile = fullfile(opts.viddir, sprintf('week11_%s_%s_%s_720p.mp4', ...
    lower(string(terrain_of(S))), lower(string(weatherStr)), lower(string(modeStr))));
if opts.makeVideo
    vw = VideoWriter(vidfile,'MPEG-4'); vw.FrameRate=round(1/dt); vw.Quality=92; open(vw);
end

fig = figure('Color','w','Units','pixels','Position',[80 80 760 820], ...
             'MenuBar','none','ToolBar','none','Renderer','opengl');
axMain = axes('Parent',fig,'Units','pixels','Position',[20 100 720 720]); hold(axMain,'on');
axis(axMain,'equal'); xlim(axMain,[xyMin(1) xyMax(1)]); ylim(axMain,[xyMin(2) xyMax(2)]);
colors = lines(max(NU,4));

% ---------- main loop ----------
for k=1:Nsteps
    t=(k-1)*dt;

    % Wind update (random but reproducible via runSeed)
    w_gust = w_gust + (-w_gust/GUST_TAU)*dt + sqrt(2*GUST_SIGMA^2/GUST_TAU)*sqrt(dt)*randn(1,2);
    w_vec = wind_at(t);

    % Rebuild THIN and THICK (static only)
    occ_thin=false(mapRows,mapCols);
    for p=1:numel(staticPolys)
        P=staticPolys{p}; occ_thin = occ_thin | inpolygon(XC,YC,P(:,1),P(:,2));
    end
    r_thick = max(1, ceil((min_clearance + 0.5*gridRes)/gridRes));
    occ_thick = imdilate(occ_thin, strel('disk', r_thick, 0));

    % Distance map from THIN (for gradients)
    distMap = imgaussfilt(bwdist(occ_thin)*gridRes, 1.5);
    [gy,gx] = gradient(distMap,gridRes,gridRes);

    % -------- Build "reserved corridors" from other UAVs to force unique paths --------
    occ_corridors = false(mapRows,mapCols);
    corridor_rad = 0.8;                                     % meters — width to discourage following same route
    dil_str = strel('disk', max(1,ceil(corridor_rad/gridRes)), 0);
    for j=1:NU
        if isempty(smoothPath{j}) || size(smoothPath{j},1)<2, continue; end
        cells = rasterize_polyline_cells(smoothPath{j}, xyMin, gridRes, mapRows, mapCols);
        if isempty(cells), continue; end
        mask = false(mapRows,mapCols);
        mask(sub2ind([mapRows,mapCols], cells(:,1), cells(:,2))) = true;
        occ_corridors = occ_corridors | imdilate(mask, dil_str);
    end

    for i=1:NU
        if reached(i)
            pos(i,:)=goals(i,:); vel(i,:)=[0 0];
            traj(k,:,i)=pos(i,:); speed_hist(k,i)=0; continue;
        end

        % If inside THICK, push out
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
        pos(i,1)=clamp(pos(i,1),xyMin(1),xyMax(1));
        pos(i,2)=clamp(pos(i,2),xyMin(2),xyMax(2));

        % Other-UAV keep-out during planning
        R_sep_plan=1.0; uav_dilate = strel('disk', max(1,ceil(R_sep_plan/gridRes)), 0);
        occ_uav_i=false(mapRows,mapCols);
        for j=1:NU
            if j==i, continue; end
            rcU=rc_from_xy(pos(j,:));
            if any(rcU<1) || rcU(1)>mapRows || rcU(2)>mapCols, continue; end
            mask=false(mapRows,mapCols); mask(rcU(1),rcU(2))=true;
            occ_uav_i = occ_uav_i | imdilate(mask, uav_dilate);
        end

        % Reserve corridors from other UAVs (do not include i's own)
        occ_corr_i = occ_corridors;
        if ~isempty(smoothPath{i}) && size(smoothPath{i},1)>=2
            cells_i = rasterize_polyline_cells(smoothPath{i}, xyMin, gridRes, mapRows, mapCols);
            if ~isempty(cells_i)
                mask_i=false(mapRows,mapCols);
                mask_i(sub2ind([mapRows,mapCols], cells_i(:,1), cells_i(:,2)))=true;
                mask_i = imdilate(mask_i, dil_str);
                occ_corr_i(mask_i) = false;
            end
        end

        occ_plan = occ_thick | occ_uav_i | occ_corr_i;

        % Periodic replan (A* + strict string-pull against THICK + corridor reserve)
        if t>=astar_t_next(i) || isempty(pathXY{i})
            astar_t_next(i)=t+ASTAR_PERIOD; vel(i,:)=[0 0];
            sRC=rc_from_xy(pos(i,:)); gRC=rc_from_xy(goals(i,:));
            occ_plan(sRC(1),sRC(2))=false; occ_plan(gRC(1),gRC(2))=false;

            w_hat = W_BASE / max(norm(W_BASE), 1e-9);
            w_wt  = WIND_WEIGHT_ASTAR;

            % Try with strong corridor reservation first, then weaken if needed
            found=false; Prc=[]; occ_try = occ_plan;
            weaken = [1.0, 0.7, 0.4, 0.0];
            for wi = 1:numel(weaken)
                if wi>1
                    occ_corr_weak = imerode(occ_corr_i, strel('disk', max(1,ceil(weaken(wi)*corridor_rad/gridRes)), 0));
                    occ_try = occ_thick | occ_uav_i | occ_corr_weak;
                    occ_try(sRC(1),sRC(2))=false; occ_try(gRC(1),gRC(2))=false;
                end
                [found,Prc] = astar_plan_clearpref(sRC,gRC,occ_try,neighbors,stepCost,distMap,safetyMargin, w_hat, w_wt);
                if found, break; end
            end
            if ~found
                occ_retry = occ_thick | occ_uav_i; occ_retry(sRC(1),sRC(2))=false; occ_retry(gRC(1),gRC(2))=false;
                [found,Prc] = astar_plan_clearpref(sRC,gRC,occ_retry,neighbors,stepCost,distMap,safetyMargin, w_hat, w_wt);
            end

            if found, P=grid2world(Prc); else, P=[pos(i,:); goals(i,:)]; end
            pathXY{i}=P;

            % strict string-pull: stop at first blocked segment (THICK)
            sp=pos(i,:); sm=sp;
            for ii2=2:size(P,1)
                if ~segFree(sp, P(ii2,:), occ_thick), break; end
                sm(end+1,:)=P(ii2,:); sp=P(ii2,:);
            end
            if size(sm,1)<2, sm=[pos(i,:); goals(i,:)]; end
            smoothPath{i}=sm;

            % THICK-aware shortcutting
            sm2 = shortcut_polyline_thick(smoothPath{i}, occ_thick, segFree, 140);
            if size(sm2,1) >= 2, smoothPath{i} = sm2; end

            replans(i)=replans(i)+1;
        end

        % Lookahead target
        seglen=sqrt(sum(diff(smoothPath{i}).^2,2)); s_cum=[0; cumsum(seglen)];
        [cp_pt,cp_idx,cp_s] = closest_point_on_polyline(pos(i,:), smoothPath{i}, s_cum);
        cursor_s(i)=max(cursor_s(i),cp_s-cursor_margin);
        s_target=cursor_s(i)+lookahead_m;

        wind_mag = norm(w_vec);
        if wind_mag > 0.7
            s_target = cursor_s(i) + (0.6 + 0.4*exp(-1.2*wind_mag)) * lookahead_m;
        end

        if s_target>=s_cum(end)
            target_wp=smoothPath{i}(end,:);
        else
            j=find(s_cum>=s_target,1);
            lam=(s_target-s_cum(j-1))/max(s_cum(j)-s_cum(j-1),1e-9);
            target_wp=smoothPath{i}(j-1,:)+lam*(smoothPath{i}(j,:)-smoothPath{i}(j-1,:));
        end
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

        % Frenet + soft repulsion
        A_=smoothPath{i}(max(cp_idx-1,1),:); B_=smoothPath{i}(min(cp_idx,size(smoothPath{i},1)),:);
        t_hat=(B_-A_)/max(norm(B_-A_),1e-9); n_hat=[-t_hat(2), t_hat(1)];
        e_y=dot(pos(i,:)-cp_pt,n_hat);
        K_lat=1.1 + 0.1*(strcmpi(modeStr,'ml')>0);
        u_path=t_hat - K_lat*e_y*n_hat;
        v_wp=(target_wp-pos(i,:)); v_wp=v_wp/max(norm(v_wp),1e-9);
        u_path=0.9*u_path+0.1*v_wp; u_path=u_path/max(norm(u_path),1e-9);

        % --- Terminal capture near goal to prevent orbiting ---
        Rcap = 2.2; Rhard = 0.8; dg = norm(pos(i,:) - goals(i,:));
        if dg < Rcap
            s_target = cursor_s(i) + max(0.6, 0.25 + 0.35*dg/Rcap) * lookahead_m;
            g_hat = (goals(i,:) - pos(i,:)) / max(dg,1e-9);
            lat_comp = u_path - dot(u_path, g_hat) * g_hat;
            damp = iff(dg < Rhard, 0.75, 0.45);
            u_path = (u_path - damp * lat_comp);
            u_path = u_path / max(norm(u_path), 1e-9);
        end

        % Obstacle-aware blend
        rc = rc_from_xy(pos(i,:));
        g_obs=[gx(rc(1),rc(2)), gy(rc(1),rc(2))]; g_obs=g_obs/max(norm(g_obs),1e-9);
        d_here=distMap(rc(1),rc(2)); d0=0.9; d1=0.3;
        if d_here>=d0, w_rep=0; elseif d_here<=d1, w_rep=1; else, sL=(d0-d_here)/(d0-d1); w_rep=sL*(2-sL); end
        side_sign=sign(e_y); if side_sign==0, side_sign=1; end
        t_slide = (side_sign>=0)*[ n_hat(2), -n_hat(1) ] + (side_sign<0)*[ -n_hat(2),  n_hat(1) ];
        t_slide=t_slide/max(norm(t_slide),1e-9);
        v_rep=0.6*g_obs + 0.6*t_slide; v_rep=v_rep/max(norm(v_rep),1e-9);
        alpha=min(ALPHA_MAX, w_rep*0.8);
        u_nom=(1-alpha)*u_path + alpha*v_rep; u_nom=u_nom/max(norm(u_nom),1e-9);

        % Heading filter
        HEADING_RATE_LIMIT=deg2rad(60); BETA=0.4;
        if norm(u_nom)>1e-9 && norm(prev_u(i,:))>1e-9
            ang_u=atan2(u_nom(2),u_nom(1)); ang_p=atan2(prev_u(i,2),prev_u(i,1));
            dAng=atan2(sin(ang_u-ang_p),cos(ang_u-ang_p));
            ang_new=ang_p + sign(dAng)*min(abs(dAng), HEADING_RATE_LIMIT*dt);
            u_nom=[cos(ang_new) sin(ang_new)];
        end
        u_sm=(1-BETA)*prev_u(i,:) + BETA*u_nom; u_sm=u_sm/max(norm(u_sm),1e-9); prev_u(i,:)=u_sm;

        % Braking rays (THICK)
        dmin=ray_max;
        for kk2=1:8
            dir=[cos(dirs(kk2)), sin(dirs(kk2))];
            s=0; hit=ray_max;
            while s<=ray_max
                q=pos(i,:)+s*dir;
                if ~segFree(pos(i,:), q, occ_thick), hit=s; break; end
                s=s+RAY_STEP;
            end
            if hit<dmin, dmin=hit; end
        end
        spd_des = vmax * (dmin>=EMERGENCY_R) + vmax * BRAKE_GAIN * max(dmin/EMERGENCY_R,0) * (dmin<EMERGENCY_R);
        spd_des = max(spd_des, 0.25*vmax);

        % -------- Wind controller (Week-10 style: simple & robust upwind) --------
        % Uses only ground-progress floor, no integral, no tacking, no lateral penalties.
        % Desired ground speed along the current heading
        distGoal = norm(pos(i,:)-goals(i,:));
        mgp = MIN_GROUND_PROGRESS_FRAC * vmax * ...
              clamp(distGoal/(3*tolReach), 0, 1);   % ground-progress floor
        v_gnd_des = max(spd_des, mgp) * u_sm;       % ground vector you want
        v_air_cmd = v_gnd_des - w_vec;              % air needed to achieve it

        % Cap airspeed
        v_air_max = SPD_AIR_MAX_FACTOR * vmax;
        na = norm(v_air_cmd);
        if na > v_air_max
            v_air_cmd = (v_air_max/na) * v_air_cmd;
        end

        % If we still don’t meet the ground-progress floor, push along-path only
        gnd_prog = dot(v_air_cmd + w_vec, u_sm);
        if gnd_prog < mgp
            v_air_cmd = v_air_cmd + (mgp - gnd_prog)*u_sm;
            na = norm(v_air_cmd);
            if na > v_air_max
                v_air_cmd = (v_air_max/na) * v_air_cmd;
            end
        end

        % Smooth acceleration limit
        dv=v_air_cmd-vel(i,:);
        dv_max=amax*dt; nv=norm(dv); if nv>dv_max, dv=(dv_max/nv)*dv; end
        vel(i,:)=vel(i,:)+dv;

        % ---- Integrate with collision bisection and tangent sliding (THICK) ----
        tryStep = (vel(i,:) + w_vec) * dt;
        Nsub    = max(1, ceil(norm(tryStep)/INT_MAX_STEP));
        sub_dt  = dt / Nsub;
        subStep = tryStep / Nsub;

        p_prev = pos(i,:);
        for ss = 1:Nsub
            p_from = pos(i,:); p_to = p_from + subStep;
            if segFree(p_from, p_to, occ_thick)
                pos(i,:) = p_to;
            else
                a = p_from; b = p_to;
                for bis = 1:18
                    m = 0.5*(a+b);
                    if segFree(p_from, m, occ_thick)
                        a = m;
                    else
                        b = m;
                    end
                end
                p_hit = a; pos(i,:) = p_hit;

                rc2 = rc_from_xy(pos(i,:));
                n2  = [gx(rc2(1),rc2(2)) gy(rc2(1),rc2(2))];
                if norm(n2) < 1e-9, n2 = [-u_sm(2), u_sm(1)]; end
                n2  = n2 / max(norm(n2),1e-9);
                tR  = [ n2(2), -n2(1)]; tL = [-n2(2),  n2(1)];
                gdir = (last_wp(i,:) - pos(i,:)); gdir = gdir / max(norm(gdir),1e-9);
                t_sl = tR; if dot(tL,gdir) > dot(tR,gdir), t_sl = tL; end
                v_slide = dot(vel(i,:), t_sl) * t_sl;
                pos_try = pos(i,:) + 0.9 * v_slide * sub_dt;
                if ~segFree(pos(i,:), pos_try, occ_thick)
                    pos_try = pos(i,:) + 0.2*gridRes * (-n2);
                end
                pos(i,:) = pos_try;
            end
            pos(i,1) = clamp(pos(i,1), xyMin(1), xyMax(1));
            pos(i,2) = clamp(pos(i,2), xyMin(2), xyMax(2));
        end

        path_len(i) = path_len(i) + norm(pos(i,:) - p_prev);

        rcH = rc_from_xy(pos(i,:)); min_clear_all(i)=min(min_clear_all(i), distMap(rcH(1),rcH(2)));

        if norm(pos(i,:)-goals(i,:))<=tolReach
            reached(i)=true; vel(i,:)=[0 0]; pos(i,:)=goals(i,:);
        end

        traj(k,:,i)=pos(i,:); speed_hist(k,i)=norm(vel(i,:));
    end

    if all(reached), break; end

    % ---------- DRAW ----------
    cla(axMain); hold(axMain,'on'); axis(axMain,'equal');
    xlim(axMain,[xyMin(1) xyMax(1)]); ylim(axMain,[xyMin(2) xyMax(2)]);

    for p=1:numel(staticPolys)
        P=staticPolys{p};
        patch('Parent',axMain,'XData',P(:,1),'YData',P(:,2), ...
              'FaceColor',polyColors{p},'FaceAlpha',0.55,'EdgeColor','k');
    end

    legH=[]; legL={};
    for i=1:opts.NU
        if ~isempty(pathXY{i}) && size(pathXY{i},1)>1
            hA=plot(axMain, pathXY{i}(:,1),pathXY{i}(:,2),'--','Color',[0.6 0.6 0.6],'LineWidth',1.2);
            legH(end+1)=hA; legL{end+1}=sprintf('A* U%d',i);
        end
        if ~isempty(smoothPath{i}) && size(smoothPath{i},1)>1
            hS=plot(axMain, smoothPath{i}(:,1),smoothPath{i}(:,2),'-','Color',[0.35 0.35 0.35],'LineWidth',1.8);
            legH(end+1)=hS; legL{end+1}=sprintf('Smooth U%d',i);
        end
        col=colors(i,:);
        hT=plot(axMain, traj(1:k,1,i),traj(1:k,2,i),'-','Color',col,'LineWidth',2.0);
        hSg=plot(axMain, starts(i,1),starts(i,2),'o','MarkerFaceColor',col,'Color',col,'MarkerSize',6);
        hG=plot(axMain, goals(i,1),goals(i,2),'x','Color',col,'LineWidth',1.6,'MarkerSize',10);
        legH=[legH hT hSg hG]; legL=[legL {sprintf('U%d traj',i) sprintf('U%d start',i) sprintf('U%d goal',i)}]; %#ok<AGROW>
    end
    legend(axMain,legH,legL,'Location','southoutside','Orientation','horizontal','AutoUpdate','off');
    title(axMain, sprintf('Week 11 (%s/%s/%s) | t=%.1fs | |wind|=%.2f m/s', ...
        upper(string(terrain_of(S))), upper(string(weatherStr)), upper(string(modeStr)), (k-1)*dt, norm(w_vec)));

    % Wind inset (captured in video)
    draw_wind_inset_on_axes(axMain, w_vec);

    % ---- write 720p frame ----
    if exist('vw','var')
        fr = getframe(axMain);
        [C,~] = frame2im(fr);
        C = imresize(C, [TARGET_HW(1) TARGET_HW(2)]);
        writeVideo(vw, C);
    end
end

if exist('vw','var'), close(vw); fprintf('Saved: %s\n', vidfile); end

lastStep = find(all(~isnan(traj(:,1,1)),2),1,'last'); if isempty(lastStep), lastStep=Nsteps; end
R.reached = reached(:)';
R.time_to_goal = (lastStep-1)*dt;
R.total_path_len = sum(path_len);
R.min_clear = min(min_clear_all(~isinf(min_clear_all))); if isempty(R.min_clear), R.min_clear=NaN; end
R.mean_speed = mean(speed_hist(~isnan(speed_hist)));
R.replan_count = sum(replans);
end

% ======================================================================
% A*
function [found,pathRC] = astar_plan_clearpref(sRC,gRC,occ_plan,neighbors,stepCost,distMap,safetyMargin, w_hat, wind_wt)
[mapRows,mapCols] = size(occ_plan);
gScore=inf(mapRows,mapCols); fScore=inf(mapRows,mapCols);
cameFrom=zeros(mapRows,mapCols,2,'int32'); openSet=false(mapRows,mapCols);
heur=@(r,c) hypot(double(r-gRC(1)),double(c-gRC(2)));
gScore(sRC(1),sRC(2))=0; fScore(sRC(1),sRC(2))=heur(sRC(1),sRC(2)); openSet(sRC(1),sRC(2))=true;
found=false; pathRC=[];
for it=1:16000
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
        clear_softmin = safetyMargin + 0.16; clear_pref = 0.80;
        dclr = distMap(rr,cc);
        if dclr<clear_softmin
            tentative = tentative + clear_pref*(1/max(dclr,1e-3) - 1/clear_softmin);
        end
        % Wind penalty: (often zero in gusty/storm via profile)
        step_vec = [double(neighbors(nb,2)), double(neighbors(nb,1))]; % [dx, dy]
        step_unit = step_vec / max(norm(step_vec), 1e-9);
        into_wind = max(0, dot(w_hat, step_unit));
        tentative = tentative + wind_wt * into_wind * stepCost(nb);

        if tentative < gScore(rr,cc)-1e-12
            cameFrom(rr,cc,:)=int32([r c]);
            gScore(rr,cc)=tentative; fScore(rr,cc)=tentative+heur(rr,cc);
            openSet(rr,cc)=true;
        end
    end
end
end

% ======================================================================
% Closest point on polyline
function [Q,idx,cp_s] = closest_point_on_polyline(p, poly, s_cum)
best_d=inf; Q=poly(1,:); idx=1; cp_s=0;
for ii=2:size(poly,1)
    A=poly(ii-1,:); B=poly(ii,:); AB=B-A;
    L=max(norm(AB),1e-9);
    tproj=max(0,min(1, dot(p-A,AB)/max(dot(AB,AB),1e-9)));
    P=A+tproj*AB; d=norm(p-P);
    if d<best_d, best_d=d; Q=P; idx=ii; cp_s=s_cum(ii-1)+tproj*L; end
end
end

% ======================================================================
% Deterministic SG placement (with BFS connectivity on inflated grid)
function [starts, goals, ok] = place_sg_random_safely_adaptive(NU, xyMin, xyMax, mapRows, mapCols, gridRes, ...
        staticPolys, XC, YC, SG_MIN_CLR, minSGsep, se_static, occ_static, cornerBoxes)

clrMap = bwdist(occ_static)*gridRes;
starts = nan(NU,2); goals = nan(NU,2); ok=false;

maxSamples   = 18000;
edgePad      = 2.0;
minPairDist  = 6.0;

% Keep a polyshape list of existing static polys to check overlap where needed
statShapes = cellfun(@(P) polyshape(P(:,1),P(:,2)), staticPolys, 'uni', 0);

si = 1;
for t=1:maxSamples
    switch si
        case 1
            candS = uniform_box(xyMin, xyMax, edgePad);
            candG = uniform_box(xyMin, xyMax, edgePad);
        case 2
            candS = edge_biased(xyMin, xyMax, edgePad);
            candG = uniform_box(xyMin, xyMax, edgePad);
        otherwise
            [candS, candG] = corner_box(cornerBoxes);
    end
    si = si + 1; if si>3, si=1; end

    if norm(candS-candG) < minPairDist, continue; end

    sRC=[ floor((candS(2)-xyMin(2))/gridRes)+1, floor((candS(1)-xyMin(1))/gridRes)+1 ];
    gRC=[ floor((candG(2)-xyMin(2))/gridRes)+1, floor((candG(1)-xyMin(1))/gridRes)+1 ];
    if any(sRC<1) || any(gRC<1) || sRC(1)>mapRows || gRC(1)>mapRows || sRC(2)>mapCols || gRC(2)>mapCols
        continue;
    end
    if ~(clrMap(sRC(1),sRC(2))>SG_MIN_CLR && clrMap(gRC(1),gRC(2))>SG_MIN_CLR)
        continue;
    end
    if ~has_path_bfs(sRC, gRC, ~occ_static)
        continue;
    end

    goodS=true; goodG=true;
    for j=1:NU
        if ~isnan(starts(j,1)), goodS = goodS && norm(candS-starts(j,:))>minSGsep; end
        if ~isnan(goals(j,1)),  goodG = goodG && norm(candG-goals(j,:))>minSGsep;  end
    end

    % Keep S/G out of static shapes (avoid on-top-of-building)
    ps = polyshape(candS(1)+[-0.05 0.05 0.05 -0.05], candS(2)+[-0.05 -0.05 0.05 0.05]);
    pg = polyshape(candG(1)+[-0.05 0.05 0.05 -0.05], candG(2)+[-0.05 -0.05 0.05 0.05]);
    for sih=1:numel(statShapes)
        if overlaps(ps,statShapes{sih}) || overlaps(pg,statShapes{sih})
            goodS=false; goodG=false; break;
        end
    end

    if goodS && goodG
        ii=find(isnan(starts(:,1)),1);
        starts(ii,:)=candS; goals(ii,:)=candG;
    end
    if all(~isnan(starts(:,1))) && all(~isnan(goals(:,1))), ok=true; break; end

    if mod(t,3000)==0
        [candS, candG] = corner_box(cornerBoxes);
        sRC=[ floor((candS(2)-xyMin(2))/gridRes)+1, floor((candS(1)-xyMin(1))/gridRes)+1 ];
        gRC=[ floor((candG(2)-xyMin(2))/gridRes)+1, floor((candG(1)-xyMin(1))/gridRes)+1 ];
        if all(sRC>=1) && sRC(1)<=mapRows && sRC(2)<=mapCols && ...
           all(gRC>=1) && gRC(1)<=mapRows && gRC(2)<=mapCols && ...
           clrMap(sRC(1),sRC(2))>0.5*SG_MIN_CLR && clrMap(gRC(1),gRC(2))>0.5*SG_MIN_CLR && ...
           has_path_bfs(sRC, gRC, ~occ_static)

            ii=find(isnan(starts(:,1)),1); if ~isempty(ii), starts(ii,:)=candS; end
            jj=find(isnan(goals(:,1)),1);  if ~isempty(jj), goals(jj,:)=candG;  end
            if all(~isnan(starts(:,1))) && all(~isnan(goals(:,1))), ok=true; break; end
        end
    end
end

    function tf = overlaps(a,b)
        tf = false;
        try
            tf = area(intersect(a,b)) > 1e-6;
        catch
            tf = false;
        end
    end
    function p = uniform_box(xyMin_, xyMax_, pad_)
        p = [xyMin_(1)+pad_+(xyMax_(1)-xyMin_(1)-2*pad_)*rand, ...
             xyMin_(2)+pad_+(xyMax_(2)-xyMin_(2)-2*pad_)*rand];
    end
    function p = edge_biased(xyMin_, xyMax_, pad_)
        side = randi(4);
        switch side
            case 1, p = [xyMin_(1)+pad_, xyMin_(2)+pad_ + (xyMax_(2)-xyMin_(2)-2*pad_)*rand];
            case 2, p = [xyMax_(1)-pad_, xyMin_(2)+pad_ + (xyMax_(2)-xyMin_(2)-2*pad_)*rand];
            case 3, p = [xyMin_(1)+pad_ + (xyMax_(1)-xyMin_(1)-2*pad_)*rand, xyMin_(2)+pad_];
            otherwise, p = [xyMin_(1)+pad_ + (xyMax_(1)-xyMin_(1)-2*pad_)*rand, xyMax_(2)-pad_];
        end
    end
    function [ps, pg] = corner_box(boxes)
        i = randi(4);
        j = mod(i+1,4)+1;
        ps = [boxes(i,1)+boxes(i,3)*rand, boxes(i,2)+boxes(i,4)*rand];
        pg = [boxes(j,1)+boxes(j,3)*rand, boxes(j,2)+boxes(j,4)*rand];
    end
end

% ---------- fast BFS on boolean free-mask ----------
function ok = has_path_bfs(sRC, gRC, freeMask)
[H,W] = size(freeMask);
if ~freeMask(sRC(1),sRC(2)) || ~freeMask(gRC(1),gRC(2)) 
    ok=false; 
    return; 
end
%#ok<*NOPRT>
vis = false(H,W);
Q = zeros(H*W,2,'uint32'); qh=1; qt=1;
Q(qt,:) = uint32(sRC); qt=qt+1; vis(sRC(1),sRC(2))=true;
nbrs = int32([ -1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1 ]);
while qh<qt
    cur = Q(qh,:); qh=qh+1;
    if cur(1)==gRC(1) && cur(2)==gRC(2), ok=true; return; end
    for k=1:8
        r = int32(cur(1))+nbrs(k,1); c = int32(cur(2))+nbrs(k,2);
        if r<1 || r>H || c<1 || c>W, continue; end
        if ~freeMask(r,c) || vis(r,c), continue; end
        if abs(nbrs(k,1))==1 && abs(nbrs(k,2))==1
            if ~freeMask(int32(cur(1)),c) || ~freeMask(r,int32(cur(2)))
                continue;
            end
        end
        vis(r,c)=true; Q(qt,:) = [r c]; qt=qt+1;
    end
end
ok=false;
end

% ======================================================================
function [staticPolys, polyColors, pondInfo] = build_terrain(name, xyMin, xyMax)
name = lower(string(name));
switch name
    case "urban",    [staticPolys, polyColors, pondInfo] = make_urban(xyMin, xyMax);
    case "natural",  [staticPolys, polyColors, pondInfo] = make_natural(xyMin, xyMax);
    case "man_made", [staticPolys, polyColors, pondInfo] = make_manmade(xyMin, xyMax);
    otherwise, error('Unknown terrain "%s".', name);
end
end

function [P,C,pondInfo] = make_urban(xyMin, xyMax)
P={}; C={}; pondInfo=[];
span = xyMax - xyMin;
xmid=mean([xyMin(1) xyMax(1)]);
ymid=mean([xyMin(2) xyMax(2)]);
nx = randi([5 6]); ny = randi([5 6]);
for ix=1:nx
for iy=1:ny
    cx = xyMin(1)+(ix-0.5)*span(1)/nx;
    cy = xyMin(2)+(iy-0.5)*span(2)/ny;
    if abs(cx-xmid)<1.6 || abs(cy-ymid)<1.6, continue; end
    if rand < 0.12, continue; end
    w=0.8+1.4*rand; h=0.8+1.4*rand; th=0.2*pi*randn;
    rect = oriented_rect([cx cy], w, h, th);
    P{end+1}=rect; C{end+1}=[0.45 0.45 0.5];
end
end
end

function [P,C,pondInfo] = make_natural(xyMin, xyMax)
% Natural: central pond; keep all other features OUT of pond and non-overlapping
P={}; C={};
cx=0; cy=0; rx=3; ry=2; th=0;
pond = oriented_ellipse_poly([cx cy],rx,ry,th,64);
P{end+1}=pond; C{end+1}=[0.4 0.7 0.9];    % water
pondInfo = struct('cx',cx,'cy',cy,'rx',rx,'ry',ry);

inside_ellipse = @(x,y,cx_,cy_,rx_,ry_,buf) (((x-cx_).^2)/(rx_+buf)^2 + ((y-cy_).^2)/(ry_+buf)^2) <= 1;

% Rock/log rectangles away from pond, non-overlapping
shore_buf = 0.30;
nRect = 4; attempts=0; added=0;
rectShapes = {}; rectColors = {};
while added<nRect && attempts<400
    attempts=attempts+1;
    L=3.5+rand*2.0; w=0.55+0.45*rand;
    c=[xyMin(1)+2+(xyMax(1)-xyMin(1)-4)*rand, xyMin(2)+2+(xyMax(2)-xyMin(2)-4)*rand];
    th=pi*rand;
    rect=oriented_rect(c,L,w,th);

    verts = rect;
    mids = 0.5*(rect + rect([2 3 4 1],:));
    pts  = [verts; mids];
    if any(inside_ellipse(pts(:,1),pts(:,2),cx,cy,rx,ry,shore_buf))
        continue;
    end

    ps = polyshape(rect(:,1), rect(:,2));
    ov=false;
    for q=1:numel(rectShapes)
        try
            if area(intersect(ps, rectShapes{q})) > 1e-6, ov=true; break; end
        catch
        end
    end
    if ov, continue; end

    rectShapes{end+1}=ps; rectColors{end+1}=[0.6 0.6 0.7]; %#ok<AGROW>
    added=added+1;
end
for q=1:numel(rectShapes)
    P{end+1}=[rectShapes{q}.Vertices; rectShapes{q}.Vertices(1,:)]; C{end+1}=rectColors{q};
end

% Trees as small discs (green dots) — ensure centers are OUTSIDE pond + margin
nTrees = 36; tries=0; placed=0;
while placed<nTrees && tries<3000
    tries=tries+1;
    tx=xyMin(1)+(xyMax(1)-xyMin(1))*rand;
    ty=xyMin(2)+(xyMax(2)-xyMin(2))*rand;
    r=0.2+0.35*rand;
    if inside_ellipse(tx,ty,cx,cy,rx,ry,shore_buf + r)
        continue;
    end
    circ=oriented_ellipse_poly([tx ty],r,r,0,14);
    P{end+1}=circ; C{end+1}=[0.2 0.5 0.2];
    placed=placed+1;
end
end

function [P,C,pondInfo] = make_manmade(xyMin, xyMax)
P={}; C={}; pondInfo=[];
span=xyMax-xyMin;
for i=1:6
    if mod(i,2)==0
        y0=xyMin(2)+(i-0.5)*span(2)/6;
        rect=oriented_rect([0 y0], span(1)*0.9, 0.8, 0);
        P{end+1}=rect; C{end+1}=[0.7 0.8 0.6];
    end
end
harbor=[xyMin(1) xyMin(2);
        xyMin(1)+0.25*span(1) xyMin(2);
        xyMin(1)+0.25*span(1) xyMax(2);
        xyMin(1) xyMax(2)];
P{end+1}=harbor; C{end+1}=[0.4 0.7 0.9];
for i=1:2
    cx=-2+4*i; cy=0;
    rect=oriented_rect([cx cy],span(1)*0.9,0.15,pi/8*randn);
    P{end+1}=rect; C{end+1}=[0.3 0.3 0.4];
end
end

function R = oriented_rect(center,w,h,th)
cx=center(1); cy=center(2); hw=w/2; hh=h/2;
pts=[-hw -hh; hw -hh; hw hh; -hw hh];
c=cos(th); s=sin(th); Rot=[c -s; s c];
R=(pts*Rot')+[cx cy];
end

function E = oriented_ellipse_poly(center,rx,ry,th,N)
cx=center(1); cy=center(2);
ang=linspace(0,2*pi,N+1)'; ang(end)=[];
pts=[rx*cos(ang), ry*sin(ang)];
c=cos(th); s=sin(th); Rot=[c -s; s c];
E=(pts*Rot')+[cx cy];
end

% ---------- Wind profile (planning + control) ----------
function P = wind_profile(name)
switch lower(name)
    case 'calm'
        W_BASE=[0 0]; GUST_SIGMA=0.05; GUST_TAU=2.5;
        SPD_AIR_MAX_FACTOR=1.8;
        MIN_GROUND_PROGRESS_FRAC=0.22;
        WIND_THRUST_BOOST=0.40;
        WIND_WEIGHT_ASTAR=0.00;
        Kwind_I=0.10;
    case 'breeze'
        W_BASE=[0.35 0.00]; GUST_SIGMA=0.12; GUST_TAU=1.8;
        SPD_AIR_MAX_FACTOR=2.0;
        MIN_GROUND_PROGRESS_FRAC=0.28;
        WIND_THRUST_BOOST=0.55;
        WIND_WEIGHT_ASTAR=0.08;
        Kwind_I=0.12;
    case 'gusty'
        W_BASE=[0.60 0.10]; GUST_SIGMA=0.25; GUST_TAU=1.0;
        SPD_AIR_MAX_FACTOR=2.8;
        MIN_GROUND_PROGRESS_FRAC=0.36;  % slightly lower to avoid stalls
        WIND_THRUST_BOOST=0.95;         % more thrust reserve
        WIND_WEIGHT_ASTAR=0.00;         % do not penalize upwind moves
        Kwind_I=0.18;                   % stronger integral
    case 'storm'
        W_BASE=[0.90 0.25]; GUST_SIGMA=0.45; GUST_TAU=0.6;
        SPD_AIR_MAX_FACTOR=3.4;         % allow more airspeed authority
        MIN_GROUND_PROGRESS_FRAC=0.44;  % slightly lower than before
        WIND_THRUST_BOOST=1.25;         % more boost if progress sags
        WIND_WEIGHT_ASTAR=0.00;         % do not penalize upwind moves
        Kwind_I=0.22;                   % stronger integral
    otherwise
        W_BASE=[0 0]; GUST_SIGMA=0.1; GUST_TAU=1.5;
        SPD_AIR_MAX_FACTOR=1.9;
        MIN_GROUND_PROGRESS_FRAC=0.25;
        WIND_THRUST_BOOST=0.50;
        WIND_WEIGHT_ASTAR=0.06;
        Kwind_I=0.12;
end
P = struct('W_BASE',W_BASE,'GUST_SIGMA',GUST_SIGMA,'GUST_TAU',GUST_TAU, ...
           'SPD_AIR_MAX_FACTOR',SPD_AIR_MAX_FACTOR, ...
           'MIN_GROUND_PROGRESS_FRAC',MIN_GROUND_PROGRESS_FRAC, ...
           'WIND_THRUST_BOOST',WIND_THRUST_BOOST, ...
           'WIND_WEIGHT_ASTAR',WIND_WEIGHT_ASTAR, ...
           'Kwind_I',Kwind_I);
end

% ---------- Small plotting helpers ----------
function plot_success_bars(T, outdir)
% Makes TWO figures:
%   success_by_terrain_baseline.png
%   success_by_terrain_ml.png
[BASELINE_COLOR, ML_COLOR] = mode_colors();

terr = unique(T.terrain,'stable');  % preserve input order
baseRate = nan(numel(terr),1);
mlRate   = nan(numel(terr),1);

for i = 1:numel(terr)
    mBase = (T.terrain==terr(i)) & (T.mode=="baseline");
    mML   = (T.terrain==terr(i)) & (T.mode=="machine learning");
    if any(mBase), baseRate(i) = mean(double(T.success(mBase)),'omitnan'); end
    if any(mML),   mlRate(i)   = mean(double(T.success(mML))  ,'omitnan'); end
end

% --- Baseline only ---
f1 = figure('Color','w','Position',[80 80 900 520]); clf;
b1 = bar(baseRate,'FaceColor','flat'); b1.CData = repmat(BASELINE_COLOR,numel(baseRate),1);
set(gca,'XTick',1:numel(terr),'XTickLabel',cellstr(terr), ...
        'FontSize',11,'XAxisLocation','top');   % terrain labels at TOP
ylabel('Success rate');
title('Success by Terrain — Baseline');
ylim([0 1]); grid on;
% value labels
for k = 1:numel(baseRate)
    if ~isnan(baseRate(k))
        text(k, baseRate(k)+0.02, sprintf('%.0f%%',100*baseRate(k)), ...
            'HorizontalAlignment','center','FontSize',9);
    end
end
saveas(f1, fullfile(outdir,'success_by_terrain_baseline.png'));

% --- Machine Learning only ---
f2 = figure('Color','w','Position',[80 80 900 520]); clf;
b2 = bar(mlRate,'FaceColor','flat'); b2.CData = repmat(ML_COLOR,numel(mlRate),1);
set(gca,'XTick',1:numel(terr),'XTickLabel',cellstr(terr), ...
        'FontSize',11,'XAxisLocation','top');   % terrain labels at TOP
ylabel('Success rate');
title('Success by Terrain — Machine Learning');
ylim([0 1]); grid on;
% value labels
for k = 1:numel(mlRate)
    if ~isnan(mlRate(k))
        text(k, mlRate(k)+0.02, sprintf('%.0f%%',100*mlRate(k)), ...
            'HorizontalAlignment','center','FontSize',9);
    end
end
saveas(f2, fullfile(outdir,'success_by_terrain_ml.png'));
end


function plot_box_time_len(T, outdir)
% Box plots with fixed blue/red colors for modes
[BASELINE_COLOR, ML_COLOR] = mode_colors();

% Canonical pretty labels, but we’ll force colors explicitly
modePretty = strings(height(T),1);
modePretty(T.mode=="baseline")           = "Baseline";
modePretty(T.mode=="machine learning")   = "Machine Learning";

f = figure('Color','w','Position',[80 80 1080 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% ---- Time by mode ----
nexttile;
bc1 = boxchart(categorical(modePretty), T.time_to_goal);
% MATLAB draws one BoxChart object per category value; enforce colors:
cats = categories(categorical(modePretty));
% find indices for baseline / ml in the returned BoxChart array:
% Easiest reliable way: redraw two separate calls filtered by mode.
cla; hold on;
bcB = boxchart(categorical(modePretty(T.mode=="baseline")), T.time_to_goal(T.mode=="baseline"));
bcM = boxchart(categorical(modePretty(T.mode=="machine learning")), T.time_to_goal(T.mode=="machine learning"));
bcB.BoxFaceColor = BASELINE_COLOR;
bcM.BoxFaceColor = ML_COLOR;
ylabel('Time (s)'); title('Time by mode');
grid on;

% ---- Path length by mode ----
nexttile;
cla; hold on;
bpB = boxchart(categorical(modePretty(T.mode=="baseline")), T.path_len(T.mode=="baseline"));
bpM = boxchart(categorical(modePretty(T.mode=="machine learning")), T.path_len(T.mode=="machine learning"));
bpB.BoxFaceColor = BASELINE_COLOR;
bpM.BoxFaceColor = ML_COLOR;
ylabel('Path length (m)'); title('Path length by mode');
grid on;

saveas(f, fullfile(outdir,'time_len_by_mode.png'));
end

function plot_clearance_cdf(T, outdir)
% Fixed blue/red lines for the two modes
[BASELINE_COLOR, ML_COLOR] = mode_colors();

f = figure('Color','w','Position',[80 80 640 520]); hold on;

plotCDF("Baseline"        , T.min_clear(T.mode=="baseline"),         BASELINE_COLOR);
plotCDF("Machine Learning", T.min_clear(T.mode=="machine learning"), ML_COLOR);

grid on; xlabel('Minimum clearance (m)'); ylabel('CDF');
legend('Location','southeast'); title('Clearance CDF');
saveas(f, fullfile(outdir,'clearance_cdf.png'));

    function plotCDF(lbl, x, col)
        x = x(~isnan(x));
        if isempty(x), return; end
        x = sort(x(:)); y = (1:numel(x))/numel(x);
        plot(x,y,'LineWidth',1.8,'DisplayName',lbl,'Color',col);
    end
end


% ---------- Wind inset overlay ----------
function draw_wind_inset_on_axes(ax, w_vec)
hold(ax, 'on');
xl = xlim(ax); yl = ylim(ax);
sx = xl(2)-xl(1); sy = yl(2)-yl(1);
s  = 0.22*min(sx,sy);
pad= 0.04*min(sx,sy);
x0 = xl(1) + pad;
y0 = yl(2) - pad - s;
xc = x0 + s/2;
yc = y0 + s/2;

rectangle(ax,'Position',[x0 y0 s s],'Curvature',0.06, ...
          'EdgeColor',[0.3 0.3 0.3],'LineWidth',0.8,'FaceColor','none');

plot(ax,[x0+0.1*s, x0+0.9*s],[yc yc],':','Color',[0.7 0.7 0.7],'LineWidth',0.8);
plot(ax,[xc xc],[y0+0.1*s, y0+0.9*s],':','Color',[0.7 0.7 0.7],'LineWidth',0.8);

wmag = norm(w_vec); dir = [1 0]; if wmag>=1e-9, dir = w_vec / wmag; end
L = 0.42 * s * min(1, wmag/2.0);

quiver(ax, xc, yc, L*dir(1), L*dir(2), 0, ...
       'LineWidth',1.6,'MaxHeadSize',2,'Color',[0 0 0]);

text(ax, x0+0.04*s, y0+0.04*s, sprintf('%.2f m/s', wmag), ...
     'FontSize',8, 'FontWeight','bold', 'HorizontalAlignment','left', ...
     'VerticalAlignment','bottom', 'Color',[0 0 0]);
end

% ---------- utilities ----------
function out = iff(cond,a,b)
if cond, out=a; else, out=b; end
end

function name = terrain_of(~)
name = "scenario";
end

function seed = str2seed(varargin)
acc = uint32(2166136261); % FNV offset basis
prime = uint32(16777619);
for k=1:nargin
    s = uint8([char(varargin{k}) '|']);
    for b = 1:numel(s)
        acc = bitxor(acc, uint32(s(b)));
        acc = uint32(mod(uint64(acc) * uint64(prime), 2^32));
    end
end
seed = double(bitand(acc, uint32(2^31-2)) + 1);
end

function P2 = shortcut_polyline_thick(P, occ_thick, segFree, maxIters)
if nargin<4, maxIters = 80; end
P2 = P;
if size(P2,1) <= 2, return; end
it = 0;
while it < maxIters
    it = it + 1;
    n = size(P2,1);
    if n <= 2, break; end
    imax = n - 2;
    if imax < 1, break; end
    i = randi([1, imax]);
    jmin = i + 2;
    if jmin > n, continue; end
    j = randi([jmin, n]);
    if segFree(P2(i,:), P2(j,:), occ_thick)
        P2 = [P2(1:i,:); P2(j:end,:)];
    end
end
end

function cells = rasterize_polyline_cells(P, xyMin, gridRes, H, W)
% Convert polyline points P (Nx2) into grid cell indices [r c] along the path.
cells = [];
if isempty(P) || size(P,1)<2, return; end
for k=2:size(P,1)
    a=P(k-1,:); b=P(k,:);
    segLen = max(1, ceil(norm(b-a)/(0.5*gridRes)));
    for s=0:segLen
        q = a + (s/segLen)*(b-a);
        r = floor((q(2)-xyMin(2))/gridRes)+1;
        c = floor((q(1)-xyMin(1))/gridRes)+1;
        if r>=1 && r<=H && c>=1 && c<=W
            cells(end+1,:) = [r c]; %#ok<AGROW>
        end
    end
end
if ~isempty(cells)
    cells = unique(cells,'rows');
end
end

function [Pout, Cout] = add_block_between_points(Pin, Cin, pondInfo, pA, pB, varargin)
% Add a single rectangular obstacle roughly centered between pA (start) and pB (goal),
% oriented PERPENDICULAR to the pA->pB direction. Ensures it stays out of the pond.
% Name-Value:
%   longLen (default 3.0), shortLen (0.6), clearStart (1.0), clearGoal (1.0)
prm = inputParser;
prm.addParameter('longLen', 3.0);
prm.addParameter('shortLen', 0.60);
prm.addParameter('clearStart', 1.0);
prm.addParameter('clearGoal', 1.0);
prm.parse(varargin{:});
L = prm.Results.longLen; W = prm.Results.shortLen;
cS = prm.Results.clearStart; %#ok<NASGU>
cG = prm.Results.clearGoal;  %#ok<NASGU>

Pout = Pin; Cout = Cin;

% Midpoint and perpendicular orientation
d = pB - pA; nd = norm(d);
if nd < 1e-6, return; % degenerate, do nothing
end
phi = atan2(d(2), d(1)) + pi/2; % perpendicular to path
mid = 0.5*(pA + pB);

% Build candidate rectangle
rect = oriented_rect(mid, L, W, phi);

% If any vertex is inside the pond + small buffer, nudge it outward
cx = pondInfo.cx; cy = pondInfo.cy; rx = pondInfo.rx; ry = pondInfo.ry;
inside_ellipse = @(x,y,buf) (((x-cx).^2)/(rx+buf)^2 + ((y-cy).^2)/(ry+buf)^2) <= 1;
if any( inside_ellipse(rect(:,1), rect(:,2), 0.10) )
    away = (mid - [cx cy]); na = norm(away); if na < 1e-9, away = [1 0]; na = 1; end
    shift = 0.6 * max(rx, ry);
    mid   = mid + (shift/na)*away;
    rect  = oriented_rect(mid, L, W, phi);
end

% Final: append polygon (close loop) and color
Pout{end+1} = [rect; rect(1,:)];
Cout{end+1} = [0.55 0.55 0.6]; % grey rock/log
end

function [BASELINE_COLOR, ML_COLOR] = mode_colors()
% Fixed mode colors used everywhere
BASELINE_COLOR = [0.00 0.35 0.95];   % blue
ML_COLOR       = [0.90 0.10 0.10];   % red
end
