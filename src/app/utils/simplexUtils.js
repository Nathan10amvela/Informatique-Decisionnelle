// app/utils/simplexUtils.js

export function isOptimalPrimal(tableau, epsilon = 1e-10) {
    const lastRow = tableau[tableau.length - 1];
    for (let i = 0; i < lastRow.length - 1; i++) { // Exclure RHS
        if (lastRow[i] < -epsilon) return false;
    }
    return true;
}

export function findEnteringVariablePrimal(tableau, epsilon = 1e-10) {
    const lastRow = tableau[tableau.length - 1];
    let minValue = -epsilon; // Cherche le plus négatif, mais doit être < 0
    let enteringVar = -1;
    for (let i = 0; i < lastRow.length - 1; i++) {
        if (lastRow[i] < minValue) {
            minValue = lastRow[i];
            enteringVar = i;
        }
    }
    return enteringVar;
}

export function findLeavingVariableRowPrimal(tableau, numProblemConstraints, enteringVarCol, epsilon = 1e-10) {
    let minRatio = Infinity;
    let leavingVarRow = -1;
    for (let i = 0; i < numProblemConstraints; i++) {
        const pivotCandidate = tableau[i][enteringVarCol];
        if (pivotCandidate > epsilon) { // Pivot doit être > 0
            const ratio = tableau[i][tableau[0].length - 1] / pivotCandidate;
            if (ratio >= -epsilon && ratio < minRatio) { // Ratio non-négatif (>=0)
                minRatio = ratio;
                leavingVarRow = i;
            } else if (Math.abs(ratio - minRatio) < epsilon && ratio >= -epsilon) {
                // Règle de Bland optionnelle ici pour les ratios égaux
            }
        }
    }
    return leavingVarRow;
}

export function pivot(tableau, pivotRow, pivotCol, epsilon = 1e-10) {
    const pivotElement = tableau[pivotRow][pivotCol];
    if (Math.abs(pivotElement) < epsilon) {
        throw new Error(`Élément pivot (${pivotElement}) proche de zéro à [${pivotRow},${pivotCol}].`);
    }
    const numCols = tableau[pivotRow].length;
    for (let j = 0; j < numCols; j++) {
        tableau[pivotRow][j] /= pivotElement;
    }
    tableau[pivotRow][pivotCol] = 1.0; // Assurer la précision

    for (let i = 0; i < tableau.length; i++) {
        if (i !== pivotRow) {
            const factor = tableau[i][pivotCol];
            for (let j = 0; j < numCols; j++) {
                tableau[i][j] -= factor * tableau[pivotRow][j];
            }
            tableau[i][pivotCol] = 0.0; // Assurer la précision
        }
    }
}

export function copyTableau(tableau) {
    return tableau.map(row => [...row]);
}

// Version simplifiée : retourne toujours la valeur Z du tableau (qui est la valeur de la fonction minimisée par le tableau)
// L'inversion pour la maximisation originale se fait dans la fonction appelante.
export function extractSolutionPrimal(tableau, numOriginalVars, basicVarsInRows, epsilon = 1e-10) {
    const solutionVariables = Array(numOriginalVars).fill(0);
    for (let i = 0; i < basicVarsInRows.length; i++) {
        const varIndexInTableau = basicVarsInRows[i];
        if (varIndexInTableau < numOriginalVars) { // Variable de décision originale
            let value = tableau[i][tableau[0].length - 1];
            solutionVariables[varIndexInTableau] = Math.abs(value) < epsilon ? 0 : value;
        }
    }
    
    let objectiveValue = tableau[tableau.length - 1][tableau[0].length - 1];
    if (Math.abs(objectiveValue) < epsilon) objectiveValue = 0;

    return {
        variables: solutionVariables,
        objectiveValue: objectiveValue // C'est la valeur de l'objectif que le tableau a minimisé
    };
}