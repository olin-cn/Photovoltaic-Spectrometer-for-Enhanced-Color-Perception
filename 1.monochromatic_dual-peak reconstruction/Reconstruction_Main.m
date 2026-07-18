clearvars; clc;

packageDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(packageDir, 'data');
outDir = fullfile(packageDir, 'outputs');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

MeasuredSignals1 = readmatrix(fullfile(dataDir, 'MeasuredSignals.xlsx'), 'Sheet', 'Fig3c');
MeasuredSignals2 = readmatrix(fullfile(dataDir, 'MeasuredSignals.xlsx'), 'Sheet', 'Fig3h');
Fig3cReferenceRaw = readmatrix(fullfile(dataDir, 'Fig3cSpectra.xlsx'), 'Sheet', 'ReferenceSpectra');
Fig3cWavelength = Fig3cReferenceRaw(:, 1);
Fig3cReferenceSpectra = Fig3cReferenceRaw(:, 2:end);
Fig3hReferenceRaw = readmatrix(fullfile(dataDir, 'Fig3hSpectra.xlsx'), 'Sheet', 'ReferenceSpectra');
Fig3hWavelength = Fig3hReferenceRaw(:, 1);
Fig3hReferenceSpectra = Fig3hReferenceRaw(:, 2);

%% Obtain models from original RMatrix files
NumModels = 2;
for i = 1:NumModels
    K1 = table2array(readtable(fullfile(dataDir, ['ResponseMatrix_', num2str(i), '.xlsx'])));
    Models(i).Kernel = K1(2:end, 2:end) / max(max(K1(2:end, 2:end)));
    Models(i).Wavelength = K1(2:end, 1);
    Models(i).Kernel = expand(Models(i).Kernel, 641);
    Models(i).Wavelength = expand(Models(i).Wavelength, 641);
    ModelsConfig(i).NumSensors = size(Models(i).Kernel, 2);
    ModelsConfig(i).NumDataPoints = length(Models(i).Wavelength);
    ModelsConfig(i).WavelengthRange = [Models(i).Wavelength(1), Models(i).Wavelength(end)];
    ModelsConfig(i).dWavelength = ModelsConfig(i).WavelengthRange(2) - ModelsConfig(i).WavelengthRange(1);
end

%% Fig. 3c: load MeasuredSignals1 and spectra
NumSpectra3c = size(MeasuredSignals1, 1);
Data3c(NumSpectra3c) = struct('Wavelength', [], 'Spectrum', [], 'Peaks', []);
for i = 1:NumSpectra3c
    Data3c(i).Wavelength = Fig3cWavelength;
    Data3c(i).Spectrum = Fig3cReferenceSpectra(:, i);
    Data3c(i).Peaks = [];
end
for SolverId = 1:NumSpectra3c
    SolverConfig3c = MakeGaussianDictionarySolver(Models(1).Wavelength, 641, 10.0, 1e-5);
    [Solution3c(SolverId), Measurement3c(SolverId)] = GetSpectraData( ...
        Models(1), SolverConfig3c, Data3c(SolverId), MeasuredSignals1(SolverId, :));
    Fig3cReconstructed(:, SolverId) = NormalizeColumn(Solution3c(SolverId).Spectrum);
end
Fig3cWavelengthOut = Solution3c(1).Wavelength;
Fig3cReferenceOut = interp1(Fig3cWavelength, Fig3cReferenceSpectra, Fig3cWavelengthOut, 'pchip', 0);
Fig3cCurveTable = table(Fig3cWavelengthOut, Fig3cReconstructed,  ...
    'VariableNames', {'Wavelength_nm', 'Reconstructed'});
writetable(Fig3cCurveTable, fullfile(outDir, 'Fig3c_Reconstruction.csv'));
PlotFig3cLocal(Fig3cWavelengthOut, Fig3cReconstructed, Fig3cReferenceOut, fullfile(outDir, 'Fig3c_Reconstruction.png'));

%% Fig. 3h: load MeasuredSignals2 and spectra
Data3h(1) = struct('Wavelength', Fig3hWavelength, ...
    'Spectrum', Fig3hReferenceSpectra, 'Peaks', []);
SolverConfig3h = MakeGaussianDictionarySolver(Models(2).Wavelength, 641, 0.5, 1e-4);
[Solution3h, Measurement3h] = GetSpectraData(Models(2), SolverConfig3h, Data3h(1), MeasuredSignals2(1, :));
Fig3hReconstructed = NormalizeColumn(Solution3h.Spectrum);
Fig3hWavelengthOut = Solution3h.Wavelength;
Fig3hCurveTable = table(Fig3hWavelengthOut, Fig3hReconstructed,  ...
    'VariableNames', {'Wavelength_nm', 'Reconstructed'});
writetable(Fig3hCurveTable, fullfile(outDir, 'Fig3h_Reconstruction.csv'));
PlotFig3hLocal(Fig3hWavelengthOut, Fig3hReconstructed, Fig3hWavelength, ...
    Fig3hReferenceSpectra, fullfile(outDir, 'Fig3h_Reconstruction.png'));

%% Save results
save(fullfile(outDir, 'ReproductionResults_vdwRS2_loaded.mat'), ...
    'Solution3c', 'Measurement3c', 'Fig3cReconstructed',  ...
    'Solution3h', 'Measurement3h', 'Fig3hReconstructed',  '-v7.3');
fprintf('Wrote outputs to: %s\n', outDir);

%% Functions
function SolverConfig = MakeGaussianDictionarySolver(wavelength, basisNum, basisFwhmNm, secondOrderPenalty)
    SolverConfig.BasisNum = basisNum;
    SolverConfig.BasisFWHM = basisFwhmNm;
    SolverConfig.nonnegative = true;
    SolverConfig.IsBatch = false;
    SolverConfig.regularized = true;
    SolverConfig.intrule = 'basis';
    SolverConfig.BasisFunction = @(x) EvalFunctionHandle(x, ...
        @(x, mu) GetGaussianPeak(x, mu, SolverConfig.BasisFWHM), ...
        linspace(wavelength(1), wavelength(end), SolverConfig.BasisNum));
    SolverConfig.BasisMatrix = SolverConfig.BasisFunction(wavelength);
    SolverConfig.alpha = 0;
    SolverConfig.beta = [0, secondOrderPenalty];
end

function y = NormalizeColumn(y)
    y = y(:);
    if max(y) > 0
        y = y ./ max(y);
    end
end

function metrics = ComputeResponseMetricsLocal(model, solution, measuredSignals)
    if isvector(solution), solution = solution(:); end
    metrics = zeros(numel(solution), 3);
    for k = 1:numel(solution)
        spectrum = interp1(solution(k).Wavelength, solution(k).Spectrum, model.Wavelength, 'pchip', 0);
        predicted = zeros(1, size(model.Kernel, 2));
        for j = 1:size(model.Kernel, 2)
            predicted(j) = trapz(model.Wavelength, spectrum .* model.Kernel(:, j));
        end
        predicted = NormalizeRow(predicted);
        measured = NormalizeRow(measuredSignals(k, :));
        err = predicted - measured;
        metrics(k, 1) = sqrt(mean(err .^ 2));
        metrics(k, 2) = max(abs(err));
        c = corrcoef(predicted, measured);
        metrics(k, 3) = c(1, 2);
    end
end

function y = NormalizeRow(y)
    y = y(:).';
    if max(abs(y)) > 0
        y = y ./ max(abs(y));
    end
end

function PlotFig3cLocal(wavelength, reconstructed, reference, outPath)
    fig = figure('Color', 'white', 'Position', [100 100 800 656], 'ToolBar', 'none', 'MenuBar', 'none');
    ax = axes('Parent', fig, 'Position', [0.17 0.18 0.78 0.78]); hold(ax, 'on');
    try, disableDefaultInteractivity(ax); catch, end
    colors = [0.33 0.33 0.95; 0.13 0.70 0.55; 0.50 0.75 0.15; 0.72 0.46 0.18; 0.82 0.20 0.25; 0.45 0.22 0.20];
    for i = 1:size(reconstructed, 2)
        if i == 1
            plot(ax, wavelength, reconstructed(:, i), '-', 'Color', colors(i, :), 'LineWidth', 1.8, 'DisplayName', 'Reconstructed');
        else
            plot(ax, wavelength, reconstructed(:, i), '-', 'Color', colors(i, :), 'LineWidth', 1.8, 'HandleVisibility', 'off');
        end
    end
    for i = 1:size(reference, 2)
        if i == 1
            plot(ax, wavelength, NormalizeColumn(reference(:, i)), ':', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.9, 'DisplayName', 'References');
        else
            plot(ax, wavelength, NormalizeColumn(reference(:, i)), ':', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.9, 'HandleVisibility', 'off');
        end
    end
    xlabel(ax, 'Wavelength (nm)', 'FontSize', 26);
    ylabel(ax, 'Intensity (a.u.)', 'FontSize', 26);
    xlim(ax, [360 940]); ylim(ax, [-0.02 1.20]);
    legend(ax, 'Location', 'northwest', 'Box', 'off', 'FontSize', 22);
    grid(ax, 'off'); box(ax, 'on');
    set(ax, 'FontSize', 22, 'LineWidth', 1.0, 'TickDir', 'out', 'XTick', 400:100:900, 'YTick', 0:0.2:1.2);
    print(fig, outPath, '-dpng', '-r150');
    close(fig);
end

function PlotFig3hLocal(wavelength, reconstructed, targetWavelength, measuredTarget, outPath)
    fig = figure('Color', 'white', 'Position', [160 120 620 520], 'ToolBar', 'none', 'MenuBar', 'none');
    ax = axes('Parent', fig); hold(ax, 'on');
    try, disableDefaultInteractivity(ax); catch, end
    plot(ax, wavelength, reconstructed, 'Color', [0.25 0.50 0.85], ...
        'LineWidth', 2.0, 'DisplayName', 'Reconstructed');
    plot(ax, targetWavelength, measuredTarget, '--', 'Color', [0.45 0.45 0.45], ...
        'LineWidth', 2.0, 'DisplayName', 'Reference');
    xlabel(ax, 'Wavelength (nm)'); ylabel(ax, 'Intensity (a.u.)');
    xlim(ax, [878 881]); ylim(ax, [-0.05 1.12]);
    grid(ax, 'on'); box(ax, 'on');
    legend(ax, 'Location', 'northwest');
    print(fig, outPath, '-dpng', '-r150');
    close(fig);
end

function [Solution, Measurement] = GetSpectraData(Models, SolverConfig,Data,lph)
    NumModels = length(Models);
    NumSpectra = length(Data);
    Solution(NumModels,NumSpectra) = struct('Wavelength',[],'Spectrum',[],...
        'RegParameter',[]);
    Measurement(NumModels,NumSpectra) = struct('ResponseRef',[],...
        'Response',[]);
    for i = 1:NumSpectra        
        for j = 1:NumModels
            Response = lph;
            [ti,xi] = SolveSpectrumNaive(Models(j).Wavelength, Models(j).Kernel, Response, SolverConfig);
            Solution(j,i).Wavelength = ti;
            Solution(j,i).Spectrum = xi/max(xi(:));          
            if ~SolverConfig.IsBatch
                Measurement(j,i).Response = Response;
            end
        end
    end
end

function [vi,spectrum,config] = SolveSpectrumNaive(wavelength, kernel, response, config)
    assert(iscolumn(wavelength), 'Wavelength must be a column vector');
    assert(all(diff(wavelength)>0), 'Wavelength points must be strictly increasing');
    assert(length(wavelength)==size(kernel,1), 'The dimensions of wavelength and kernel must be matched');
    assert(size(response,2)==size(kernel,2), 'The number of photoelectric responses must be the same as the number of rows in the kernel matrix');
    [A, ti, config] = GetLinearForm(wavelength, kernel, config);
    b = response';
    H = A'*A;
    f = -A'*b;
    [xi,config] = SolveLinearForm(H, f, config);
    vi = linspace(ti(1),ti(end),length(wavelength))'; 
    spectrum = interp1(ti, xi, vi,'pchip');
end

function [xi,config] = SolveLinearForm(H, f, config)    
    if config.regularized
        [alpha, beta] = GetPenaltyParameter(config);
        [H,f] = Regularize(H,f, alpha, beta);
    end
    if config.nonnegative
        y = SolveNonnegativeQPNoToolbox(H, f);
    else
        y = -pinv(H) * f;
    end
    config.BasisCoefficient = y;
    if strcmp(config.intrule,'basis')
        xi = zeros(size(config.BasisMatrix,1),1);
        for i = 1:size(config.BasisMatrix,2)
            xi = xi + config.BasisMatrix(:,i)*y(i);
        end
    else
        xi = y;
    end
    function [alpha, beta] = GetPenaltyParameter(config)
        alpha = config.alpha;
        beta = config.beta;        
    end
end

function y = SolveNonnegativeQPNoToolbox(H, f)
    n = length(f);
    y = zeros(n, 1);
    passive = false(n, 1);
    tol = 1e-12;
    maxIter = max(100, 20 * n);
    for iter = 1:maxIter
        grad = H * y + f;
        w = -grad;
        inactive = ~passive;
        if ~any(inactive) || max(w(inactive)) <= tol
            break;
        end
        candidates = find(inactive);
        [~, localIdx] = max(w(candidates));
        passive(candidates(localIdx)) = true;
        while true
            z = zeros(n, 1);
            Hp = H(passive, passive);
            fp = f(passive);
            z(passive) = -pinv(Hp) * fp;
            if all(z(passive) > tol)
                y = z;
                break;
            end
            nonpos = passive & (z <= tol);
            denom = y(nonpos) - z(nonpos);
            valid = denom > 0;
            alphaVals = y(nonpos);
            alphaVals = alphaVals(valid) ./ denom(valid);
            if isempty(alphaVals)
                alpha = 0;
            else
                alpha = min(alphaVals);
            end
            y = y + alpha * (z - y);
            remove = passive & (y <= tol);
            passive(remove) = false;
            y(remove) = 0;
            if ~any(passive)
                break;
            end
        end
    end
    y(y < 0 & y > -sqrt(eps)) = 0;
end

function Y=EvalFunctionHandle(x, FunctionHandle, StructureParams)
    NumCols = length(StructureParams);
    NumRows = length(x);
    Y = zeros(NumRows,NumCols);
    for m = 1:length(StructureParams)
        Y(:,m) = FunctionHandle(x, StructureParams(m));
    end
end

function Sp = GetGaussianPeak(xi, mu, fwhm)
    sigma = fwhm/(2*sqrt(2*log(2))); % get std from fwhm;
    Sp = exp(-0.5*((xi-mu)/sigma).^2);
end

function [Q,P] = Regularize(H,f, alpha, beta)
    Q = H; P = f;
    if ~isempty(beta)
        for order = 1:length(beta)
            if beta(order)~=0
                L = GetGradMatrix(size(f,1), order-1);
                Q = Q + beta(order)*(L'*L);
            end
        end
    end
    if ~isempty(alpha)
        if isscalar(alpha)
            P = P + alpha(1);
        else
            error('Undefined Behavior');
        end
    end    
end

function [A, ti, config] = GetLinearForm(wavelength, kernel, config)
    assert(strcmp(config.intrule, 'basis'), 'Only basis integration is used in this reproduction script.');
    [A, ti, config] = IntRuleBasis(wavelength, kernel, config);
    
end

function [A, ti, config] = IntRuleBasis(wavelength, kernel, config)
    BasisMatrix = config.BasisFunction(wavelength);
    NumBasis = size(BasisMatrix,2); % n
    NumSensors = size(kernel,2); % m
    A = zeros(NumSensors,NumBasis);
    for i = 1:NumSensors
        for j = 1:NumBasis
            A(i,j) = trapz(wavelength, kernel(:,i).*BasisMatrix(:,j));
        end
    end
    ti = wavelength;
    config.BasisMatrix = BasisMatrix;
end

function interpolatedMatrix=expand(originalMatrix,newRows)
interpolatedMatrix = zeros(newRows, size(originalMatrix, 2));
for col = 1:size(originalMatrix, 2)
    originalY = 1:size(originalMatrix, 1);   
    newY = linspace(1, size(originalMatrix, 1), newRows);    
    interpolatedMatrix(:, col) = interp1(originalY, originalMatrix(:, col), newY, 'pchip');
end
end

function L = GetGradMatrix(n, SmoothOrder)        
    if SmoothOrder == 0
        L = spdiags(ones(n,1),0,n,n);
        return
    end
    L = MakeGradMatrix1(n);
    if n > SmoothOrder
        L = L^(SmoothOrder);
        L(end-SmoothOrder+1:end,:)=[]; 
    end            
    function L0 = MakeGradMatrix1(n)
        idy = [1:n,1:n-1];
        idx = [1:n,2:n];
        v = [ones(1,n),-ones(1,n-1)];
        L0 = sparse(idy,idx,v,n,n,2*n-1);
    end
end

