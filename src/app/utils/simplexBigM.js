// app/utils/simplexBigM.js

import {
    isOptimalPrimal,
    findEnteringVariablePrimal,
    findLeavingVariableRowPrimal,
    pivot,
    copyTableau,
    extractSolutionPrimal
} from './simplexUtils'; // Ajustez le chemin si nécessaire

const M_PENALTY = 100000; // Grande valeur pour M

// Fonction principale pour résoudre un tableau Big M (commune à Max et Min)
function solveBigMTableau(tableau, basicVarsInRows, numOriginalVars, numProblemConstraints, numArtificialVarsIndices, steps, epsilon, isOriginalProblemMaximization) {
    let iteration = 0;
    const maxIterations = 100; // Augmenté pour Big M potentiellement plus lent

    while (!isOptimalPrimal(tableau, epsilon) && iteration < maxIterations) {
        iteration++;
        const enteringVarCol = findEnteringVariablePrimal(tableau, epsilon);
        if (enteringVarCol === -1) break;

        const leavingVarRow = findLeavingVariableRowPrimal(tableau, numProblemConstraints, enteringVarCol, epsilon);
        if (leavingVarRow === -1) {
            steps.push({ type: 'unbounded', message: 'Problème non borné (Big M).' });
            return { status: 'unbounded', message: 'Problème non borné (Big M).', steps };
        }
        
        const actualLeavingVar = basicVarsInRows[leavingVarRow];
        steps.push({
            type: 'pivot_info',
            message: `Itération BigM ${iteration}: Entre col ${enteringVarCol + 1}, Sort (col ${actualLeavingVar + 1}) de ligne ${leavingVarRow + 1}.`,
            entering: enteringVarCol,
            leaving_row: leavingVarRow,
            pivot_element: tableau[leavingVarRow][enteringVarCol]
        });

        pivot(tableau, leavingVarRow, enteringVarCol, epsilon);
        basicVarsInRows[leavingVarRow] = enteringVarCol;

        steps.push({
            type: 'iteration_tableau',
            message: `Tableau après itération BigM ${iteration}`,
            tableau: copyTableau(tableau),
            basicVars: [...basicVarsInRows]
        });
    }

    if (iteration >= maxIterations) {
        steps.push({ type: 'max_iterations', message: "Max itérations (Big M)." });
        return { status: 'max_iterations', message: "Max itérations (Big M).", steps };
    }

    // Vérifier si des variables artificielles sont en base avec une valeur > 0
    for (const artVarIndex of numArtificialVarsIndices) {
        const rowIndex = basicVarsInRows.indexOf(artVarIndex);
        if (rowIndex !== -1) { // Si la variable artificielle est en base
            const value = tableau[rowIndex][tableau[0].length - 1];
            if (Math.abs(value) > epsilon) {
                steps.push({ type: 'infeasible', message: `Problème infaisable: Variable artificielle (col ${artVarIndex + 1}) en base avec valeur ${value.toFixed(3)}.` });
                return { status: 'infeasible', message: `Infaisable: Var. Artificielle x${artVarIndex + 1} en base avec valeur non nulle.`, steps };
            }
        }
    }
    
    let extractedData = extractSolutionPrimal(tableau, numOriginalVars, basicVarsInRows, epsilon);
    let finalObjectiveValue = extractedData.objectiveValue;

    if (isOriginalProblemMaximization) {
        finalObjectiveValue = -finalObjectiveValue; // Car on a minimisé -Z
    }
    // Pour la minimisation, la valeur est déjà correcte.

    steps.push({
        type: 'final_solution',
        message: 'Solution optimale trouvée (Big M).',
        solutionObject: { variables: extractedData.variables, objectiveValue: finalObjectiveValue },
        finalTableau: copyTableau(tableau)
    });

    return {
        status: 'optimal',
        solution: {
            variables: extractedData.variables,
            objectiveValue: finalObjectiveValue
        },
        steps: steps
    };
}


// Fonction de base pour construire le tableau Big M
function setupBigMTableauCommon(originalObjective, originalConstraints, originalRhs, originalInequalities, isMaximization, steps, epsilon) {
    const numOriginalVars = originalObjective.length;
    const numProblemConstraints = originalConstraints.length;

    let c_tableau = isMaximization ? originalObjective.map(x => -x) : [...originalObjective];
    let A_tableau = originalConstraints.map(row => [...row]);
    let b_tableau = [...originalRhs];
    let effectiveInequalities = [...originalInequalities];

    // Standardiser RHS >= 0
    for (let i = 0; i < numProblemConstraints; i++) {
        if (b_tableau[i] < -epsilon) {
            A_tableau[i] = A_tableau[i].map(val => -val);
            b_tableau[i] = -b_tableau[i];
            if (effectiveInequalities[i] === '<=') effectiveInequalities[i] = '>=';
            else if (effectiveInequalities[i] === '>=') effectiveInequalities[i] = '<=';
            steps.push({ type: 'transformation', message: `Contrainte ${i+1} et RHS multipliés par -1 car RHS < 0.` });
        }
    }

    let numSlack = 0;
    let numSurplus = 0;
    let numArtificial = 0;
    const artificialVarOriginalIndices = []; // Pour stocker les indices de colonnes des variables artificielles

    effectiveInequalities.forEach((ineq, i) => {
        if (ineq === '<=') numSlack++;
        else if (ineq === '>=') {
            numSurplus++;
            numArtificial++;
        } else if (ineq === '=') {
            numArtificial++;
        }
    });

    const totalVarsInTableau = numOriginalVars + numSlack + numSurplus + numArtificial;
    const tableauRows = numProblemConstraints + 1;
    const tableauCols = totalVarsInTableau + 1;

    let tableau = Array(tableauRows).fill(null).map(() => Array(tableauCols).fill(0));
    let basicVarsInRows = Array(numProblemConstraints).fill(-1);

    let currentSlackCol = numOriginalVars;
    let currentSurplusCol = numOriginalVars + numSlack;
    let currentArtificialCol = numOriginalVars + numSlack + numSurplus;

    for (let i = 0; i < numProblemConstraints; i++) {
        for (let j = 0; j < numOriginalVars; j++) tableau[i][j] = A_tableau[i][j];
        tableau[i][tableauCols - 1] = b_tableau[i];

        if (effectiveInequalities[i] === '<=') {
            tableau[i][currentSlackCol] = 1;
            basicVarsInRows[i] = currentSlackCol;
            currentSlackCol++;
        } else if (effectiveInequalities[i] === '>=') {
            tableau[i][currentSurplusCol] = -1; // Variable d'excès
            // currentSurplusCol++; // On ne l'ajoute pas aux basicVars
            
            tableau[i][currentArtificialCol] = 1; // Variable artificielle
            basicVarsInRows[i] = currentArtificialCol;
            artificialVarOriginalIndices.push(currentArtificialCol);
            currentSurplusCol++; // Important: incrémenter surplus après l'avoir utilisé pour l'index de colonne
            currentArtificialCol++;
        } else if (effectiveInequalities[i] === '=') {
            tableau[i][currentArtificialCol] = 1; // Variable artificielle
            basicVarsInRows[i] = currentArtificialCol;
            artificialVarOriginalIndices.push(currentArtificialCol);
            currentArtificialCol++;
        }
    }

    // Ligne Z initiale (coûts des variables originales)
    for (let j = 0; j < numOriginalVars; j++) tableau[tableauRows - 1][j] = c_tableau[j];
    // Coûts des surplus vars (si existent) sont 0 dans la fonction Z (leur effet est via l'artificielle)

    // Ajouter pénalités M pour les variables artificielles dans la ligne Z
    artificialVarOriginalIndices.forEach(artColIdx => {
        tableau[tableauRows - 1][artColIdx] = M_PENALTY;
    });

    // Éliminer les M de la ligne Z pour les variables artificielles qui sont en base
    for (let i = 0; i < numProblemConstraints; i++) {
        const basicVarForThisRow = basicVarsInRows[i];
        if (artificialVarOriginalIndices.includes(basicVarForThisRow)) {
            // C'est une variable artificielle en base.
            // On doit faire: LigneZ = LigneZ - M * LigneContrainte_i
            // Puisque le coeff de l'artificielle dans LigneZ est M, et dans LigneContrainte_i est 1.
            const M_in_Z_for_this_art_var = tableau[tableauRows - 1][basicVarForThisRow]; // Devrait être M_PENALTY
            if (Math.abs(M_in_Z_for_this_art_var) > epsilon) { // Si M est bien là
                 for (let k = 0; k < tableauCols; k++) {
                    tableau[tableauRows - 1][k] -= M_in_Z_for_this_art_var * tableau[i][k];
                }
            }
        }
    }
    
    steps.push({ type: 'initial_tableau_bigm', message: 'Tableau initial (Big M)', tableau: copyTableau(tableau), basicVars: [...basicVarsInRows], artificialVarsCols: [...artificialVarOriginalIndices] });
    return { tableau, basicVarsInRows, numOriginalVars, numProblemConstraints, artificialVarOriginalIndices, steps };
}


export function solveBigMMaximization(originalObjective, originalConstraints, originalRhs, originalInequalities) {
    const steps = [];
    const epsilon = 1e-10;
    steps.push({ type: 'method_info', message: 'Début: Méthode du Grand M (Maximisation).' });

    const setupResult = setupBigMTableauCommon(originalObjective, originalConstraints, originalRhs, originalInequalities, true, steps, epsilon);
    if (setupResult.status === 'error') return setupResult; // Propagation d'erreur si setup échoue

    return solveBigMTableau(
        setupResult.tableau,
        setupResult.basicVarsInRows,
        setupResult.numOriginalVars,
        setupResult.numProblemConstraints,
        setupResult.artificialVarOriginalIndices,
        setupResult.steps, // steps est déjà dans setupResult
        epsilon,
        true // isOriginalProblemMaximization
    );
}

export function solveBigMMinimization(originalObjective, originalConstraints, originalRhs, originalInequalities) {
    const steps = [];
    const epsilon = 1e-10;
    steps.push({ type: 'method_info', message: 'Début: Méthode du Grand M (Minimisation).' });

    const setupResult = setupBigMTableauCommon(originalObjective, originalConstraints, originalRhs, originalInequalities, false, steps, epsilon);
     if (setupResult.status === 'error') return setupResult;

    return solveBigMTableau(
        setupResult.tableau,
        setupResult.basicVarsInRows,
        setupResult.numOriginalVars,
        setupResult.numProblemConstraints,
        setupResult.artificialVarOriginalIndices,
        setupResult.steps,
        epsilon,
        false // isOriginalProblemMaximization
    );
}