// app/utils/simplexStandard.js

// Fonctions utilitaires (pourraient être dans un autre fichier helper plus tard)
// Ces fonctions sont basées sur le tableau standard où l'objectif est de minimiser.
// Pour la maximisation, on minimise -Z.

function isOptimalInternal(tableau, epsilon = 1e-10) {
    const lastRow = tableau[tableau.length - 1];
    for (let i = 0; i < lastRow.length - 1; i++) {
        if (lastRow[i] < -epsilon) return false;
    }
    return true;
}

function findEnteringVariableInternal(tableau, epsilon = 1e-10) {
    const lastRow = tableau[tableau.length - 1];
    let minValue = -epsilon; // Doit être strictement négatif pour entrer
    let enteringVar = -1;
    for (let i = 0; i < lastRow.length - 1; i++) {
        if (lastRow[i] < minValue) {
            minValue = lastRow[i];
            enteringVar = i;
        }
    }
    return enteringVar;
}

function findLeavingVariableRowInternal(tableau, numProblemConstraints, enteringVar, epsilon = 1e-10) {
    let minRatio = Infinity;
    let leavingVarRow = -1; // Index de la ligne de contrainte
    for (let i = 0; i < numProblemConstraints; i++) {
        const pivotCandidate = tableau[i][enteringVar];
        if (pivotCandidate > epsilon) {
            const ratio = tableau[i][tableau[0].length - 1] / pivotCandidate;
            if (ratio >= 0 && ratio < minRatio) {
                minRatio = ratio;
                leavingVarRow = i;
            }
            // Gérer la règle de Bland pour les ratios égaux (simplifié ici)
            else if (Math.abs(ratio - minRatio) < epsilon && ratio >=0) {
                // Optionnel: Si basicVars est accessible, on pourrait appliquer Bland
                // Pour l'instant, on prend le premier trouvé ou on pourrait ajouter une logique de plus petit indice
            }
        }
    }
    return leavingVarRow;
}

function pivotInternal(tableau, pivotRow, pivotCol, epsilon = 1e-10) {
    const pivotElement = tableau[pivotRow][pivotCol];
    if (Math.abs(pivotElement) < epsilon) {
        throw new Error(`Élément pivot nul ou proche de zéro (${pivotElement}) à [${pivotRow}, ${pivotCol}].`);
    }
    for (let j = 0; j < tableau[pivotRow].length; j++) {
        tableau[pivotRow][j] /= pivotElement;
    }
    tableau[pivotRow][pivotCol] = 1.0; // Forcer à 1.0 pour précision

    for (let i = 0; i < tableau.length; i++) {
        if (i !== pivotRow) {
            const factor = tableau[i][pivotCol];
            for (let j = 0; j < tableau[i].length; j++) {
                tableau[i][j] -= factor * tableau[pivotRow][j];
            }
            tableau[i][pivotCol] = 0.0; // Forcer à 0.0 pour précision
        }
    }
}

function extractSolutionInternal(tableau, numOriginalVars, isMaximizationProblem, epsilon = 1e-10, basicVarsInRows) {
    const solutionVariables = Array(numOriginalVars).fill(0);
    // basicVarsInRows[i] = k signifie que la variable k (colonne k) est basique dans la ligne i

    for (let i = 0; i < basicVarsInRows.length; i++) { // basicVarsInRows a la longueur du nombre de contraintes
        const varIndexInTableau = basicVarsInRows[i]; // Colonne de la var de base pour la ligne i
        if (varIndexInTableau < numOriginalVars) { // Si c'est une variable de décision originale
            solutionVariables[varIndexInTableau] = tableau[i][tableau[0].length - 1]; // Sa valeur est le RHS
        }
    }
    
    let objectiveValue = tableau[tableau.length - 1][tableau[0].length - 1];
    if (isMaximizationProblem) {
        objectiveValue = -objectiveValue; // Car on a minimisé -Z
    }
    
    const cleanedSolutionVariables = solutionVariables.map(val => Math.abs(val) < epsilon ? 0 : val);
    if (Math.abs(objectiveValue) < epsilon && objectiveValue !== 0) {
        objectiveValue = 0;
    }

    return {
        variables: cleanedSolutionVariables,
        objectiveValue: objectiveValue
    };
}

function copyTableauInternal(tableau) {
    return tableau.map(row => [...row]);
}


// --- Fonction principale pour la méthode Standard Maximisation ---
export function solveStandardMaximization(originalObjective, originalConstraints, originalRhs, originalInequalities) {
    const steps = [];
    const numOriginalVars = originalObjective.length;
    const numProblemConstraints = originalConstraints.length;
    const epsilon = 1e-10;

    steps.push({
        type: 'method_info',
        message: 'Début de la méthode Simplexe Standard (Maximisation).'
    });

    // 1. Vérification des conditions pour la méthode standard
    //    (Toutes contraintes <=, tous RHS >= 0)
    for (let i = 0; i < numProblemConstraints; i++) {
        if (originalInequalities[i] !== '<=') {
            steps.push({ type: 'error', message: `Contrainte ${i+1} n'est pas '<='. La méthode standard simple requiert des contraintes '<='.` });
            return { status: 'error', message: `Méthode standard non applicable: contrainte ${i+1} non '<='.` , steps };
        }
        if (originalRhs[i] < -epsilon) { // Permettre une petite marge pour 0
             steps.push({ type: 'error', message: `RHS de la contrainte ${i+1} est négatif. La méthode standard simple requiert des RHS >= 0.` });
            return { status: 'error', message: `Méthode standard non applicable: RHS de contrainte ${i+1} < 0.` , steps };
        }
    }
    
    // 2. Préparation du tableau
    // Pour maximisation, on minimise -Z. Donc, c = -originalObjective.
    const c_tableau = originalObjective.map(x => -x);
    const A_tableau = originalConstraints.map(row => [...row]); // Copie
    const b_tableau = [...originalRhs]; // Copie

    // Nombre de variables d'écart = nombre de contraintes
    const numSlackVars = numProblemConstraints;
    const totalVarsInTableau = numOriginalVars + numSlackVars;
    
    const tableauRows = numProblemConstraints + 1; // Contraintes + ligne Z
    const tableauCols = totalVarsInTableau + 1;    // Variables + RHS

    let tableau = Array(tableauRows).fill(null).map(() => Array(tableauCols).fill(0));
    let basicVarsInRows = Array(numProblemConstraints).fill(-1); // Stocke l'index de la variable de base pour chaque ligne de contrainte

    // Remplir les contraintes et variables d'écart
    for (let i = 0; i < numProblemConstraints; i++) {
        // Coeffs des variables originales
        for (let j = 0; j < numOriginalVars; j++) {
            tableau[i][j] = A_tableau[i][j];
        }
        // Variable d'écart pour cette contrainte
        const slackVarCol = numOriginalVars + i;
        tableau[i][slackVarCol] = 1;
        basicVarsInRows[i] = slackVarCol; // La variable d'écart i est en base pour la ligne i
        // RHS
        tableau[i][tableauCols - 1] = b_tableau[i];
    }

    // Remplir la ligne Z (fonction objectif)
    for (let j = 0; j < numOriginalVars; j++) {
        tableau[tableauRows - 1][j] = c_tableau[j];
    }
    // Les variables d'écart ont un coût de 0 dans la fonction Z.
    // La valeur de Z est initialement 0.

    steps.push({
        type: 'initial_tableau',
        message: 'Tableau initial (Standard Maximisation)',
        tableau: copyTableauInternal(tableau),
        basicVars: [...basicVarsInRows] // Pourrait être utile pour l'affichage
    });

    // 3. Itérations du Simplexe
    let iteration = 0;
    const maxIterations = 50; // Sécurité

    while (!isOptimalInternal(tableau, epsilon) && iteration < maxIterations) {
        iteration++;
        const enteringVarCol = findEnteringVariableInternal(tableau, epsilon);
        if (enteringVarCol === -1) {
            // Normalement, isOptimalInternal aurait dû retourner true
            steps.push({ type: 'info', message: "Condition d'optimalité atteinte (pas de variable entrante négative)." });
            break;
        }

        const leavingVarRow = findLeavingVariableRowInternal(tableau, numProblemConstraints, enteringVarCol, epsilon);
        if (leavingVarRow === -1) {
            steps.push({ type: 'unbounded', message: 'Problème non borné (pas de variable sortante éligible).' });
            return { status: 'unbounded', message: 'Problème non borné.', steps };
        }

        const actualLeavingVar = basicVarsInRows[leavingVarRow]; // La variable qui sort de la base

        steps.push({
            type: 'pivot_info',
            message: `Itération ${iteration}: Variable entrante colonne ${enteringVarCol + 1}, Variable sortante (colonne ${actualLeavingVar + 1}) de la ligne ${leavingVarRow + 1}.`,
            entering: enteringVarCol,
            leaving_row: leavingVarRow,
            pivot_element: tableau[leavingVarRow][enteringVarCol]
        });

        pivotInternal(tableau, leavingVarRow, enteringVarCol, epsilon);
        basicVarsInRows[leavingVarRow] = enteringVarCol; // Mettre à jour la variable de base pour cette ligne

        steps.push({
            type: 'iteration_tableau',
            message: `Tableau après itération ${iteration}`,
            tableau: copyTableauInternal(tableau),
            basicVars: [...basicVarsInRows]
        });
    }

    if (iteration >= maxIterations) {
        steps.push({ type: 'max_iterations', message: "Nombre maximum d'itérations atteint." });
        return { status: 'max_iterations', message: "Nombre maximum d'itérations atteint.", steps };
    }

    // 4. Extraction de la solution
    const extractedData = extractSolutionInternal(tableau, numOriginalVars, true /*isMaximizationProblem*/, epsilon, basicVarsInRows);
    
    steps.push({
        type: 'final_solution',
        message: 'Solution optimale trouvée (Standard Maximisation).',
        // Pour la cohérence avec l'affichage, on peut dupliquer ici ou l'affichage devra s'adapter.
        // Pour l'instant, on garde la structure attendue par l'affichage précédent.
        solutionObject: { // Renommé pour éviter confusion avec le `solution` de l'objet retour global
             variables: extractedData.variables,
             objectiveValue: extractedData.objectiveValue
        },
        finalTableau: copyTableauInternal(tableau)
    });

    return {
        status: 'optimal',
        solution: { // <<< IMBRICATION ICI
            variables: extractedData.variables,
            objectiveValue: extractedData.objectiveValue
        },
        steps: steps
    };
}


// --- Fonction principale pour la méthode Standard Minimisation ---
// --- Fonction principale pour la méthode Standard Minimisation avec transformation Dual ---
export function solveStandardMinimization(originalObjective, originalConstraints, originalRhs, originalInequalities) {
    const steps = [];
    const numOriginalVars = originalObjective.length;
    const numProblemConstraints = originalConstraints.length;
    const epsilon = 1e-10;

    steps.push({
        type: 'method_info',
        message: 'Début de la méthode Simplexe Standard (Minimisation avec gestion des contraintes >=).'
    });

    // Vérification si le problème nécessite une transformation duale
    const hasGeqConstraints = originalInequalities.some(ineq => ineq === '>=');
    const hasNegativeRhs = originalRhs.some(rhs => rhs < -epsilon);
    
    if (hasGeqConstraints || hasNegativeRhs) {
        steps.push({
            type: 'dual_transformation_needed',
            message: 'Le problème contient des contraintes >= ou des RHS négatifs. Application de la transformation duale.'
        });
        
        return solveDualTransformation(originalObjective, originalConstraints, originalRhs, originalInequalities, steps, epsilon);
    }

    // Si pas de transformation duale nécessaire, procéder normalement
    return solveDirectMethod(originalObjective, originalConstraints, originalRhs, originalInequalities, steps, epsilon);
}

// --- Fonction pour résoudre par transformation duale ---
function solveDualTransformation(originalObjective, originalConstraints, originalRhs, originalInequalities, steps, epsilon) {
    const numOriginalVars = originalObjective.length;
    const numOriginalConstraints = originalConstraints.length;

    steps.push({
        type: 'dual_info',
        message: 'Construction du problème dual:'
    });

    // Construction du problème dual
    // Problème primal: Min cX sujet à AX >= b, X >= 0
    // Problème dual: Max bY sujet à A^T Y <= c, Y >= 0
    
    // Variables duales: une par contrainte du primal
    const dualObjective = [...originalRhs]; // b devient l'objectif dual (à maximiser)
    
    // Matrice des contraintes duales: A^T
    const dualConstraints = [];
    for (let j = 0; j < numOriginalVars; j++) {
        const dualRow = [];
        for (let i = 0; i < numOriginalConstraints; i++) {
            dualRow.push(originalConstraints[i][j]);
        }
        dualConstraints.push(dualRow);
    }
    
    // RHS dual: c (coefficients de l'objectif primal)
    const dualRhs = [...originalObjective];
    
    // Toutes les contraintes duales sont <=
    const dualInequalities = Array(numOriginalVars).fill('<=');

    steps.push({
        type: 'dual_problem',
        message: 'Problème dual construit:',
        dualObjective: [...dualObjective],
        dualConstraints: dualConstraints.map(row => [...row]),
        dualRhs: [...dualRhs],
        dualInequalities: [...dualInequalities]
    });

    // Transformation du problème dual de maximisation en minimisation
    // Max bY devient Min -bY
    const minDualObjective = dualObjective.map(coeff => -coeff);

    steps.push({
        type: 'dual_to_min',
        message: 'Transformation du dual (Max → Min): objectif multiplié par -1'
    });

    // Résolution du problème dual transformé
    const dualResult = solveDirectMethod(minDualObjective, dualConstraints, dualRhs, dualInequalities, steps, epsilon);
    
    if (dualResult.status !== 'optimal' || !dualResult.finalTableau) { // Vérifier aussi finalTableau
            steps.push({type: 'error', message: 'Le problème dual n\'a pas pu être résolu de manière optimale ou le tableau final est manquant.'});
            return dualResult; // Retourner l'erreur du dual
    }

    // Reconstruction de la solution primale à partir de la solution duale
    steps.push({
        type: 'dual_to_primal',
        message: 'Reconstruction de la solution primale à partir de la solution duale'
    });

    // La valeur objectif primale = -valeur objectif dual (car on avait inversé le signe)
    const primalObjectiveValue = -dualResult.objectiveValue;
    
    // Les variables primales correspondent aux variables d'écart duales
    // Cette reconstruction nécessite l'analyse du tableau final dual
    const primalSolution = reconstructPrimalFromDual(dualResult, originalConstraints, originalRhs, originalObjective, steps, epsilon);
    
    if (primalSolution.status !== 'reconstructed' && primalSolution.status !== 'optimal') { // Ajuster le statut attendu
        return {status: 'error', message: 'Erreur lors de la reconstruction de la solution primale.', steps};
    }

    steps.push({
        type: 'final_solution',
        message: 'Solution primale (Min cX) reconstruite avec succès.',
        // On va mettre la structure directement attendue par page.js
        // solutionObject: { 
        //     variables: primalSolutionData.variables,
        //     objectiveValue: primalObjectiveValue
        // }
    });

    return {
        status: 'optimal',
        message: 'Solution optimale trouvée via transformation duale',
        solution: { // <<< AJOUTER L'IMBRICATION ICI
            variables: primalSolution.variables,
            objectiveValue: primalObjectiveValue
        },
        steps: steps,
        dualSolution: {
            variables: dualResult.variables,
            objectiveValue: -dualResult.objectiveValue // Valeur correcte pour le dual (Max bY)
        }
    };
}

// --- Fonction pour reconstruire la solution primale à partir de la solution duale ---
/*function reconstructPrimalFromDual(dualResult, originalConstraints, originalRhs, originalObjective, steps, epsilon) {
    // Cette fonction utilise les conditions de complémentarité du théorème de dualité forte
    // Pour une implémentation complète, on devrait analyser le tableau final
    // Ici, on propose une approche simplifiée
    
    const numPrimalVars = originalObjective.length;
    const numDualVars = dualResult.variables.length;
    
    // Initialisation des variables primales
    let primalVariables = Array(numPrimalVars).fill(0);
    
    // Méthode simplifiée: utiliser les conditions de complémentarité
    // Si une contrainte duale est saturée (égalité), la variable primale correspondante peut être > 0
    
    // Pour une reconstruction plus précise, il faudrait résoudre le système:
    // A^T * y = c (pour les variables primales de base)
    // Cette implémentation nécessiterait l'accès au tableau final du dual
    
    steps.push({
        type: 'primal_reconstruction',
        message: 'Reconstruction simplifiée: variables primales initialisées à 0. Une implémentation complète nécessiterait l\'analyse du tableau dual final.'
    });
    
    return {
        status: 'optimal',
        variables: primalVariables
    };
}*/

// Dans app/utils/simplexStandard.js

function reconstructPrimalFromDual(dualSolveResult, originalNumPrimalVars, steps, epsilon) {
    // dualSolveResult est l'objet retourné par solveDirectMethod (donc pour le problème dual)
    // dualSolveResult.finalTableau est le tableau optimal du problème dual Min (-bY) s.t. A^T Y + S_dual = c
    // dualSolveResult.variables contient les valeurs des variables Y.
    // dualSolveResult.objectiveValue est la valeur de Min (-bY).

    if (!dualSolveResult.finalTableau) {
        steps.push({ type: 'error', message: 'Tableau final du dual manquant pour la reconstruction primale.' });
        return { status: 'error', variables: [] };
    }

    const dualFinalTableau = dualSolveResult.finalTableau;
    const dualObjectiveRow = dualFinalTableau[dualFinalTableau.length - 1];
    
    // Les variables primales (X_j) correspondent aux coûts réduits des variables d'écart
    // du problème dual (S_dual_j).
    // Dans le tableau dual, les variables d'écart S_dual_j commencent après les variables Y_i.
    // Le nombre de variables Y_i est `dualSolveResult.variables.length` (qui est numOriginalConstraints du primal).
    // Donc, la colonne de S_dual_1 est `numDualYVars`, S_dual_2 est `numDualYVars + 1`, etc.
    // Et il y a `numOriginalPrimalVars` variables d'écart duales (une pour chaque contrainte A^T Y <= c_j,
    // donc une pour chaque variable primale X_j).

    const numDualYVars = dualSolveResult.variables.length; // = numOriginalConstraints du primal
    let primalVariables = Array(originalNumPrimalVars).fill(0);

    for (let j = 0; j < originalNumPrimalVars; j++) {
        // La j-ième variable primale X_j correspond à la (j)-ième variable d'écart du dual S_dual_j.
        // Sa colonne dans le tableau dual est numDualYVars + j.
        const slackVarColInDualTableau = numDualYVars + j;
        if (slackVarColInDualTableau < dualObjectiveRow.length - 1) { // S'assurer qu'on ne dépasse pas
            // La valeur de X_j est le coût réduit (coefficient dans la ligne Z) de S_dual_j.
            // Puisque le tableau dual minimise (-bY), les coûts réduits sont déjà corrects pour les valeurs de X_j.
            // Si le coût réduit était pour un problème de maximisation, il faudrait prendre l'opposé.
            // Ici, c'est une minimisation, donc les Zj (coûts réduits) >= 0.
            // La valeur de la variable primale Xj est le Zj de la variable d'écart duale Sj.
            primalVariables[j] = dualObjectiveRow[slackVarColInDualTableau];
            if (Math.abs(primalVariables[j]) < epsilon) {
                primalVariables[j] = 0;
            }
        } else {
            steps.push({ type: 'warning', message: `Impossible de trouver le coût réduit pour la variable primale X_${j+1} (colonne ${slackVarColInDualTableau} hors limites).`});
        }
    }
    
    steps.push({
        type: 'primal_reconstruction_done',
        message: `Variables primales reconstruites à partir des coûts réduits du dual: [${primalVariables.map(v => v.toFixed(3)).join(', ')}]`
    });
    
    return {
        status: 'reconstructed', // ou 'optimal' si on considère que c'est la fin
        variables: primalVariables
    };
}

// --- Méthode directe (sans transformation duale) ---
function solveDirectMethod(objective, constraints, rhs, inequalities, steps, epsilon) {
    const numOriginalVars = objective.length;
    const numProblemConstraints = constraints.length;

    let c_tableau = [...objective];
    let A_tableau = constraints.map(row => [...row]);
    let b_tableau = [...rhs];
    let effectiveInequalities = [...inequalities];

    // Transformation des contraintes >= en <=
    for (let i = 0; i < numProblemConstraints; i++) {
        if (effectiveInequalities[i] === '>=') {
            A_tableau[i] = A_tableau[i].map(val => -val);
            b_tableau[i] = -b_tableau[i];
            effectiveInequalities[i] = '<=';
            steps.push({
                type: 'transformation',
                message: `Contrainte ${i+1} (>=) transformée en <= : Multipliée par -1.`
            });
        } else if (effectiveInequalities[i] === '=') {
            steps.push({ 
                type: 'error', 
                message: `Contrainte ${i+1} est '='. La méthode standard simple ne gère pas les égalités directement (utiliser Grand M).` 
            });
            return { 
                status: 'error', 
                message: `Méthode standard non applicable: contrainte ${i+1} est '='.`,
                steps 
            };
        }

        if (b_tableau[i] < -epsilon && effectiveInequalities[i] === '<=') {
            steps.push({ 
                type: 'error', 
                message: `RHS de la contrainte ${i+1} (potentiellement transformée) est négatif. Nécessite méthode duale.` 
            });
            return { 
                status: 'error', 
                message: `Méthode standard non applicable: RHS de contrainte ${i+1} < 0.`,
                steps 
            };
        }
    }

    // Construction du tableau simplex
    const numSlackVars = numProblemConstraints;
    const totalVarsInTableau = numOriginalVars + numSlackVars;
    const tableauRows = numProblemConstraints + 1;
    const tableauCols = totalVarsInTableau + 1;

    let tableau = Array(tableauRows).fill(null).map(() => Array(tableauCols).fill(0));
    let basicVarsInRows = Array(numProblemConstraints).fill(-1);

    // Remplissage du tableau
    for (let i = 0; i < numProblemConstraints; i++) {
        for (let j = 0; j < numOriginalVars; j++) {
            tableau[i][j] = A_tableau[i][j];
        }
        const slackVarCol = numOriginalVars + i;
        tableau[i][slackVarCol] = 1;
        basicVarsInRows[i] = slackVarCol;
        tableau[i][tableauCols - 1] = b_tableau[i];
    }

    // Ligne objectif
    for (let j = 0; j < numOriginalVars; j++) {
        tableau[tableauRows - 1][j] = c_tableau[j];
    }

    steps.push({
        type: 'initial_tableau',
        message: 'Tableau initial',
        tableau: copyTableauInternal(tableau),
        basicVars: [...basicVarsInRows]
    });

     // Itérations du simplexe
    const iterationResult = performSimplexIterations(tableau, basicVarsInRows, numOriginalVars, numProblemConstraints, steps, epsilon);

    // Propager le finalTableau si la résolution est optimale
    if (iterationResult.status === 'optimal') {
        return {
            ...iterationResult, // Contient status, variables, objectiveValue, steps, finalTableau
            // message: 'Solution directe optimale', // Optionnel, déjà dans les steps
        };
    }
    return iterationResult; // Retourner tel quel si ce n'est pas optimal
}

// --- Fonctions utilitaires pour les itérations du simplexe ---
function performSimplexIterations(tableau, basicVarsInRows, numOriginalVars, numConstraints, steps, epsilon) {
    let iteration = 0;
    const maxIterations = 50;

    while (!isOptimalInternal(tableau, epsilon) && iteration < maxIterations) {
        iteration++;
        const enteringVarCol = findEnteringVariableInternal(tableau, epsilon);
        if (enteringVarCol === -1) break;

        const leavingVarRow = findLeavingVariableRowInternal(tableau, numConstraints, enteringVarCol, epsilon);
        if (leavingVarRow === -1) {
            steps.push({ type: 'unbounded', message: 'Problème non borné.' });
            return { status: 'unbounded', message: 'Problème non borné.', steps };
        }
        
        const actualLeavingVar = basicVarsInRows[leavingVarRow];
        steps.push({
            type: 'pivot_info',
            message: `Itération ${iteration}: Var entrante col ${enteringVarCol + 1}, Var sortante (col ${actualLeavingVar + 1}) de ligne ${leavingVarRow + 1}.`,
            entering: enteringVarCol,
            leaving_row: leavingVarRow,
            pivot_element: tableau[leavingVarRow][enteringVarCol]
        });

        pivotInternal(tableau, leavingVarRow, enteringVarCol, epsilon);
        basicVarsInRows[leavingVarRow] = enteringVarCol;

        steps.push({
            type: 'iteration_tableau',
            message: `Tableau après itération ${iteration}`,
            tableau: copyTableauInternal(tableau),
            basicVars: [...basicVarsInRows]
        });
    }

    if (iteration >= maxIterations) {
        steps.push({ type: 'max_iterations', message: "Max itérations atteint." });
        return { status: 'max_iterations', message: "Max itérations atteint.", steps };
    }

    // Extraction de la solution finale
    const solution = extractSolutionInternal(tableau, numOriginalVars, false, epsilon, basicVarsInRows);
    
    steps.push({
        type: 'optimal_solution',
        message: 'Solution optimale trouvée (méthode directe)',
        // On ne met pas solution ici car c'est pour le problème interne résolu (le dual)
        // solutionObject: { variables: solutionData.variables, objectiveValue: solutionData.objectiveValue },
        finalTableau_direct: copyTableauInternal(tableau) // Nommer pour clart
    });

    return {
        status: 'optimal',
        message: 'Solution optimale trouvée',
        variables: solution.variables,
        objectiveValue: solution.objectiveValue,
        // message: 'Solution optimale trouvée (directe)', // Message déjà dans les steps
        // Il faut retourner les variables de CE problème résolu (les variables Y du dual)
        // et la valeur Z de CE problème résolu (Min -bY)
        finalTableau: copyTableauInternal(tableau), // <<< RETOURNER LE TABLEAU FINAL
        steps: steps

    };
}

// Note: Les fonctions utilitaires suivantes sont supposées exister dans votre code :
// - isOptimalInternal(tableau, epsilon)
// - findEnteringVariableInternal(tableau, epsilon) 
// - findLeavingVariableRowInternal(tableau, numConstraints, enteringCol, epsilon)
// - pivotInternal(tableau, pivotRow, pivotCol, epsilon)
// - copyTableauInternal(tableau)
// - extractSolutionInternal(tableau, numOriginalVars, isOriginalProblemMaximization, epsilon, basicVarsInRows)

// Si ces fonctions n'existent pas, décommentez les implémentations ci-dessous :

/*
function isOptimalInternal(tableau, epsilon) {
    const lastRow = tableau[tableau.length - 1];
    for (let j = 0; j < lastRow.length - 1; j++) {
        if (lastRow[j] < -epsilon) {
            return false;
        }
    }
    return true;
}

function findEnteringVariableInternal(tableau, epsilon) {
    const lastRow = tableau[tableau.length - 1];
    let minValue = 0;
    let enteringCol = -1;
    
    for (let j = 0; j < lastRow.length - 1; j++) {
        if (lastRow[j] < minValue) {
            minValue = lastRow[j];
            enteringCol = j;
        }
    }
    
    return enteringCol;
}

function findLeavingVariableRowInternal(tableau, numConstraints, enteringCol, epsilon) {
    let minRatio = Infinity;
    let leavingRow = -1;
    
    for (let i = 0; i < numConstraints; i++) {
        const pivotElement = tableau[i][enteringCol];
        const rhs = tableau[i][tableau[i].length - 1];
        
        if (pivotElement > epsilon) {
            const ratio = rhs / pivotElement;
            if (ratio >= -epsilon && ratio < minRatio) {
                minRatio = ratio;
                leavingRow = i;
            }
        }
    }
    
    return leavingRow;
}

function pivotInternal(tableau, pivotRow, pivotCol, epsilon) {
    const pivotElement = tableau[pivotRow][pivotCol];
    const numRows = tableau.length;
    const numCols = tableau[0].length;
    
    // Normaliser la ligne pivot
    for (let j = 0; j < numCols; j++) {
        tableau[pivotRow][j] /= pivotElement;
    }
    
    // Éliminer les autres éléments de la colonne pivot
    for (let i = 0; i < numRows; i++) {
        if (i !== pivotRow) {
            const multiplier = tableau[i][pivotCol];
            for (let j = 0; j < numCols; j++) {
                tableau[i][j] -= multiplier * tableau[pivotRow][j];
            }
        }
    }
}

function copyTableauInternal(tableau) {
    return tableau.map(row => [...row]);
}

function extractSolutionInternal(tableau, numOriginalVars, isOriginalProblemMaximization, epsilon = 1e-10, basicVarsInRows) {
    const solutionVariables = Array(numOriginalVars).fill(0);
    for (let i = 0; i < basicVarsInRows.length; i++) {
        const varIndexInTableau = basicVarsInRows[i];
        if (varIndexInTableau < numOriginalVars) {
            solutionVariables[varIndexInTableau] = tableau[i][tableau[0].length - 1];
        }
    }
    
    let objectiveValue = tableau[tableau.length - 1][tableau[0].length - 1];
    
    if (isOriginalProblemMaximization) {
        objectiveValue = -objectiveValue;
    }
    
    const cleanedSolutionVariables = solutionVariables.map(val => Math.abs(val) < epsilon ? 0 : val);
    if (Math.abs(objectiveValue) < epsilon && objectiveValue !== 0) {
        objectiveValue = 0;
    }

    return {
        variables: cleanedSolutionVariables,
        objectiveValue: objectiveValue
    };
}
*/