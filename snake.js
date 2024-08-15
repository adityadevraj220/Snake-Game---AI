var ai = 5;
// 5 is default, 2 is genetic, false means manual, true means logic AI
var exit = false;
var timer = null;
var speed = 20;
var ended = false;
var pause = false;
var totalsteps = 0;

const canvas = document.getElementById("snake");
canvas.width = screen.availWidth - 36;
canvas.height = screen.availHeight - 174;
const ctx = canvas.getContext("2d");
// x*col = canvas.width
// x*row = canvas.height
const row = 13 * 3;
const col = 30 * 3;

const wid = canvas.width / col;
const hei = canvas.height / row;

// Global variables for a single snake
var snake = [];
var target = [-1, -1];
var direction = 3;
var tempDirection = 3; // To store previous direction and avoid ambiguity while sending data for training NN
// 0 Up, 1 Down, 2 Left, 3 Right

var red = "#ff0000";
var white = "#f2f5fa";
var black = "#000000";
var yellow = "#ffff00";

//--------------------------------------------------------------------------------------------------------
// Genetic Evolution

var populationTargets = [];
var populationSize = 20;
function population(newPopBrains) {
    this.snakeNo = populationSize;
    boardColorGrid = [];
    canvas.width = canvas.width;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < row; i++) {
        var temp = [];
        for (let j = 0; j < col; j++) {
            temp.push([0, 0, 0]);
        }
        boardColorGrid.push(temp);
    }
    boardColorGrid[20][45][1] = this.snakeNo;
    boardColorGrid[20][44][0] = this.snakeNo;
    boardColorGrid[20][43][0] = this.snakeNo;
    boardColorGrid[20][42][0] = this.snakeNo;
    boardColorGrid[20][41][0] = this.snakeNo;

    this.snakes = [];

    for (let i = 0; i < this.snakeNo; i++) {
        if (newPopBrains) {
            this.snakes.push(new snakeConst(newPopBrains[i]));
        } else {
            this.snakes.push(new snakeConst());
        }
        this.snakes[i].id = i;
    }

    populationTargets = [];

    generatePopulationTarget([
        [20, 45],
        [20, 44],
        [20, 43],
        [20, 42],
        [20, 41],
    ]);

    this.promiseArr = [];

    this.run = async () => {
        for (let i = 0; i < this.snakeNo; i++) {
            this.promiseArr.push(this.snakes[i].run());
        }

        var snakePerformance = [];
        var snakeBrains = [];
        await Promise.all(this.promiseArr).then((values) => {
            console.log("end");
            values.map((item, i) => {
                // console.log("fitness of", i, item);
                snakePerformance.push(item);
                snakeBrains.push(this.snakes[i].brain);
            });
        });
        this.snakes = null;
        this.promiseArr = null;
        populationTargets = [];
        boardColorGrid = [];
        return [snakePerformance, snakeBrains];
    };
}

function snakeConst(newBrain) {
    this.fitness = 0;
    this.crashed = false;
    if (newBrain) {
        this.brain = newBrain;
    } else {
        this.brain = generateRandomNeuralNet(); // Random brain initially
    }
    //   this.fitnessValues = [];
    // row = 39
    // col = 90
    this.snake = [
        [20, 45],
        [20, 44],
        [20, 43],
        [20, 42],
        [20, 41],
    ];

    this.id = 0;
    this.run = async () => {
        var fitnessValues = await playSingleGeneticSnake(
            this.snake,
            this.brain,
            this.id
        );
        // [applesEaten, steps]
        // var fitness = (fitnessValues[0] * 1000) / fitnessValues[1] + 0.01;
        var apples = fitnessValues[0];
        var steps = fitnessValues[1];
        // var fitness = fitnessValues[0] * 100 + 1;
        var fitness =
            steps +
            (Math.pow(2, apples) + Math.pow(apples, 2.1) * 500) -
            Math.pow(apples, 1.2) * Math.pow(0.25 * steps, 1.3);
        // f(x,y) = y + 2^(x) + x^(2.1) * 500 - (x^(1.2))*((0.25 * y)^(1.3))
        // console.log(fitness);
        fitnessValues = null;
        return fitness;
    };
}

async function startGeneticEvolution() {
    var display = document.getElementById("generation");
    var species = null;
    var popData = null;
    var select = [];
    var newPopBrains = null;
    var oldBrains = [];

    document.getElementById("score").style.display = "none";
    display.style.display = "initial";

    var initialModel = await tf
        .loadLayersModel("./NeuralNetwokModel/savedModel/model.json")
        .then(async (loadedModel) => {
            console.log("loaded Model");
            return await loadedModel;
        });

    for (let i = 0; i < populationSize; i++) {
        select.push(i);
        oldBrains.push(initialModel);
    }
    newPopBrains = await generateNewGeneration(select, oldBrains);
    species = new population(newPopBrains);

    for (let i = 0; i < 10000; i++) {
        display.innerHTML = "Generation: " + (i + 1);
        popData = await species.run();
        if (exit) {
            break;
        }
        // var avgfit = 0;
        // popData[0].map((i) => (avgfit += i));
        // avgfit = avgfit / populationSize;
        select = selection(popData[0]);
        oldBrains = popData[1];
        newPopBrains = await generateNewGeneration(select, oldBrains);
        await sleep(1000);
        species = new population(newPopBrains);
        // console.log("NEW POPULATION GENERATED fitness", avgfit / 100);
    }
}

async function generateRandomNeuralNet(weights = null) {
    const model = await tf.sequential();
    await model.add(
        tf.layers.dense({
            inputShape: [24],
            activation: "sigmoid",
            units: 20,
            weights: [
                tf.randomUniform([24, 20], -2, 2),
                tf.randomUniform([20], -2, 2),
            ],
        })
    );
    await model.add(
        tf.layers.dense({
            inputShape: [20],
            activation: "sigmoid",
            units: 4,
        })
    );

    await model.compile({
        loss: "meanSquaredError",
        optimizer: tf.train.sgd(1),
    });

    await model.setWeights(weights);

    return model;
}

async function generateNewGeneration(select, oldBrains) {
    var num = oldBrains.length;
    var newPopBrains = [];
    var p1 = 0;
    var p2 = 0;
    var parent1 = null;
    var parent2 = null;
    var newBrain = null;

    for (let i = 0; i < num; i++) {
        p1 = Math.floor(Math.random() * 100) % select.length;
        p2 = Math.floor(Math.random() * 100) % select.length;
        parent1 = oldBrains[select[p1]];
        parent2 = oldBrains[select[p2]];
        newBrain = await crossoverAndMutate(parent1, parent2);
        newPopBrains.push(newBrain);
    }

    return newPopBrains;
}

async function crossoverAndMutate(parent1, parent2) {
    // p1 and p2 are Neural networks

    // Crossover
    var newGene = [];
    var gene1 = parent1.getWeights();
    var gene2 = parent2.getWeights();
    var subGene = null;
    var p1 = null;
    var p2 = null;
    for (let i = 0; i < gene1.length; i++) {
        subGene = [];
        p1 = await gene1[i].array();
        p2 = await gene2[i].array();
        if (i == 1 || i == 3) {
            // Biases
            subGene = generateRandomUnionArrayWithMutation(p1, p2);
        } else {
            // weights
            for (let i = 0; i < p1.length; i++) {
                subGene.push(
                    generateRandomUnionArrayWithMutation(p1[i], p2[i])
                );
            }
        }
        newGene.push(tf.tensor(subGene));
    }
    var newBrain = await generateRandomNeuralNet(newGene);

    return newBrain;
}

function selection(snakePerformance) {
    var max = 0;
    for (let i = 0; i < snakePerformance.length; i++) {
        if (max < snakePerformance[i]) {
            max = snakePerformance[i];
        }
    }
    for (let i = 0; i < snakePerformance.length; i++) {
        snakePerformance[i] = snakePerformance[i] / max;
    }
    var selection = [];
    for (let i = 0; i < snakePerformance.length; i++) {
        if (snakePerformance[i] == 0) {
            continue;
        }
        var t = snakePerformance[i] * 1000;
        for (let j = 0; j < t; j++) {
            selection.push(i);
        }
    }
    return shuffle(selection);
}

function generateRandomUnionArrayWithMutation(p1, p2) {
    var temp = [];
    var x = 0;
    for (let i = 0; i < p1.length; i++) {
        x = Math.floor(Math.random() * 100) % 10;
        if (x < 5) {
            temp.push(p1[i]);
        } else {
            temp.push(p2[i]);
        }

        // Mutation
        x = Math.floor(Math.random() * 100) % (100 / mutationRate); // Mutation rate is 10% (out of 10 numbers, only 1 will change) . So i took mod by by 10 and then check if the number is 1. The number will be 1 with the probablitiy of 1/10;

        if (x == 1) {
            var y = ((Math.random(0, 1) * 100) % mutationChange) / 100; // Determines how much to increase the number in the range of factor of [-1.2, 1.2]
            var changeSign = Math.floor(Math.random(0, 1) * 100) % 2; // Determines wheter to increase or decrease the number
            temp[temp.length - 1] =
                temp[temp.length - 1] * (changeSign == 0 ? 1 + y : 1 - y);
        }
    }
    return temp;
}

function shuffle(array) {
    var currentIndex = array.length,
        temporaryValue,
        randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }
    return array;
}

async function playSingleGeneticSnake(snake, brain, id) {
    var applesEaten = 0;
    var steps = 0;
    var target = populationTargets[0];
    var justRoaming = 0;

    while (true) {
        if (pause === true) {
            console.log("sleep");
            if (exit) break;
            await dopause();
            continue;
        }
        if (exit) break;
        var newHead = await getGeneticHead(snake, brain, target);
        if (newHead === null) {
            // console.log("Sneak Dead");
            return [applesEaten, steps];
        }
        snake = [[newHead[0], newHead[1]], ...snake];
        handlePopulationColorChange(newHead[0], newHead[1], red);
        handlePopulationColorChange(snake[1][0], snake[1][1], white);
        if (newHead[2] !== true) {
            var t = snake.pop();
            handlePopulationColorChange(t[0], t[1], black);
        } else {
            justRoaming = 0;
            var t = snake.pop();
            populationTargets[applesEaten][2] += 1;
            if (populationTargets[applesEaten][2] == populationSize) {
                destroyTarget(populationTargets[applesEaten]);
            }
            handlePopulationColorChange(t[0], t[1], black);
            applesEaten++;
            if (applesEaten === populationTargets.length) {
                generatePopulationTarget(snake);
                target = populationTargets[populationTargets.length - 1].slice(
                    0,
                    2
                );
            } else {
                target = populationTargets[applesEaten].slice(0, 2);
            }
        }
        steps++;

        // If the snake has not eaten the next apple in 200 steps then it dies due to starvation
        if (justRoaming > 200) {
            break;
        }
        justRoaming += 1;
        await sleep(10);
    }

    return [applesEaten, steps];
}

async function getGeneticHead(snake, brain, target) {
    // Use the Neural Network to generate a predicted direction

    // Generate the input Layer and give to the brain(Neural Network) of snake
    var inputLayer = collectNNDataSeeingAsSnake(snake, target);
    var prediction = brain.predict(tf.tensor2d(inputLayer, [1, 24]));

    var max = 0;
    return await prediction.array().then(async (array) => {
        array[0].map((x, ind) => {
            if (x > array[0][max]) {
                max = ind;
            }
        });
        // The direction with maximum probability is choosen.
        return await getNewHead(snake, max, target);
    });
}

// Getting the grid for sending data
var totalDataDirection = [];
var totalSnakeSmartNNData = [];

function collectNNDataSeeingAsSnake(snake, target) {
    // Generates the input layer data for a snake
    var inputLayer = [];

    //   -  2  - 1   -
    //     -   -   -      o                  N
    //   3   - - -     0                     |
    //    sssssS - - - - - - - -         W - O - E
    //   4   - - -     7                     |
    //     -   -   -                         S
    //   -  5  -  6  -
    // Here sssssS represents snake and S represents head of snake.
    // o represents target.
    // Sanke will look in total 8 directions, the numbers in the diagram represents the 8 sections/directions
    // For each direction three computation will be done:
    // 1. is target present in that direction.
    // 2. wall distance along the direction from the head.
    // 3. Minimum distance from head to snake's body in that direction
    // 8 * 3 = 24

    // 0 Up, 1 Down, 2 Left, 3 Right
    for (let i = 0; i < 24; i++) {
        inputLayer[i] = 0;
    }

    // 1. is target present in that direction.
    var head = [snake[0][0], snake[0][1]];
    inputLayer[getSectionFromUnformattedAngle(head, target) * 3] = 1;

    // 2. wall distance along the direction from the head.
    // This is for future, this gives details of snakes body in all 8 directions to the input layer
    // snake.forEach((val, x) => {
    //     if (x != 0) {
    //         var dis = Math.abs(val[1] - head[1]) + Math.abs(val[0] - head[0]);
    //         var section = getSectionFromUnformattedAngle(head, val);
    //         if (
    //             inputLayer[2 + section * 3] === 0 ||
    //             dis < inputLayer[2 + section * 3]
    //         ) {
    //             inputLayer[2 + section * 3] = dis / 100;
    //         }
    //     }
    // });

    // 3. Minimum distance from head to snake's body in that direction
    var root2 = 1.4142135623;
    var x = head[0];
    var y = head[1];
    inputLayer[1] = (col - 1 - y) / 100;
    inputLayer[4] =
        Math.round((col - 1 - y <= x ? col - 1 - y : x) * root2 * 10) / 1000;
    inputLayer[7] = x / 100;
    inputLayer[10] = Math.round((x >= y ? y : x) * root2 * 10) / 1000;
    inputLayer[13] = y / 100;
    inputLayer[16] =
        Math.round((row - 1 - x <= y ? row - 1 - x : y) * root2 * 10) / 1000;
    inputLayer[19] = (row - 1 - x) / 100;
    inputLayer[22] =
        Math.round(
            (row - 1 - x <= col - 1 - y ? row - 1 - x : col - 1 - y) *
                root2 *
                10
        ) / 1000;

    // Used for collecting data for training model
    if (ai == true && totalsteps > 4) {
        totalSnakeSmartNNData.push(inputLayer);
        totalDataDirection.push(tempDirection);
    }

    return inputLayer;
}

function getSectionFromUnformattedAngle(inhead, inval) {
    var head = inhead.slice();
    var val = inval.slice();
    transformXY(head);
    transformXY(val);
    var angleRad = Math.atan((val[1] - head[1]) / (val[0] - head[0]));
    var angleDeg = (angleRad * 180) / Math.PI;
    var formatAngle =
        angleDeg > 0
            ? val[0] > head[0]
                ? angleDeg
                : val[0] < head[0]
                ? 180 + angleDeg
                : 90
            : angleDeg < 0
            ? val[0] < head[0]
                ? 180 + angleDeg
                : val[0] > head[0]
                ? 270 + 90 - Math.abs(angleDeg)
                : 270
            : angleDeg == 0
            ? val[0] > head[0]
                ? 0
                : 180
            : null;

    var section =
        formatAngle < 45
            ? 0
            : formatAngle < 90
            ? 1
            : formatAngle < 135
            ? 2
            : formatAngle < 180
            ? 3
            : formatAngle < 225
            ? 4
            : formatAngle < 270
            ? 5
            : formatAngle < 315
            ? 6
            : formatAngle < 360
            ? 7
            : null;

    return section;
}

function transformXY(coord) {
    coord[0] = 39 - coord[0];
    var t = coord[0];
    coord[0] = coord[1];
    coord[1] = t;
}

var boardColorGrid = [];
function handlePopulationColorChange(x, y, color, temp = false) {
    var status = boardColorGrid[x][y];
    // console.log("status", status);
    if (color === yellow) {
        status[2] += 1;
        // console.log("YELLOW");
        changeColorAt(x, y, yellow);
    } else if (color === red) {
        if (status[2] > 0 || status[1] > 0) {
            status[1] += 1;
            // console.log("Inred1");
        } else {
            // console.log("Inred2");
            status[1] = 1;
            changeColorAt(x, y, red);
        }
    } else if (color === black) {
        if (status[0] > 1 || status[1] > 0 || status[2] > 0) {
            // console.log("INBLACK1");
        } else {
            // console.log("INBLACK2");
            changeColorAt(x, y, black);
        }
        status[0] -= 1;
    } else if (color === white) {
        // We do it when the second head is painted white
        status[1] -= 1;
        status[0] += 1;
        if (status[1] > 0) {
            // console.log("INWhite1");
        } else {
            // console.log("INWhite2");
            if (status[2] > 0) {
            } else changeColorAt(x, y, white);
        }
    }
}

function generatePopulationTarget(snake) {
    var x = Math.round((Math.random() * 1000) % (row - 1));
    var y = Math.round((Math.random() * 1000) % (col - 1));
    while (containArray(snake, [x, y])) {
        x = Math.round((Math.random() * 1000) % (row - 1));
        y = Math.round((Math.random() * 1000) % (col - 1));
    }
    handlePopulationColorChange(x, y, yellow);
    populationTargets.push([x, y, 0]);
}

function destroyTarget(targ) {
    var status = boardColorGrid[targ[0]][targ[1]];
    status[2] -= 1;
    if (status[2] > 0) {
    } else {
        if (status[1] > 0) {
            changeColorAt(targ[0], targ[1], red);
        } else {
            changeColorAt(targ[0], targ[1], white);
        }
    }
}

//--------------------------------------------------------------------------------------------------------
// Logical AI
// A*, reverse A* to solve the game

async function playAI() {
    var x = target[0];
    var y = target[1];
    console.log("Starting AI");

    while (true) {
        var sea = solve();
        // Solving the path from snake head to apple
        if (!sea) break;
        var solved = sea[0];

        // x,y is the corrdinate of the current head
        x = solved[0][0];
        y = solved[0][1];
        var i = 1;
        var end = false;
        // game[target[0]][target[1]] = 2;
        // var found = sea[1] === true ? true : false;
        while (!(x === target[0] && y === target[1])) {
            if (pause === true) {
                await dopause();
                if (exit) break;
                continue;
            }
            // 0 Up, 1 Down, 2 Left, 3 Right
            if (exit) break;
            // for pausing the game
            tempDirection = getAiDirection(x, y, snake[0][0], snake[0][1]);
            snake = [[x, y], ...snake];
            // black - 0, white - 1, yellow - 2, red - 3
            // game[x][y] = 3;
            // game[snake[1][0]][snake[1][1]] = 1;

            // adds x,y as the new head to the snake

            changeColorAt(x, y, red);
            changeColorAt(snake[1][0], snake[1][1], white);
            var t = snake.pop();
            // game[t[0]][t[1]] = 0;
            // removes the tail
            changeColorAt(t[0], t[1], black);
            // snake = snake
            // assigns the new snake
            if (solved.length === 0) {
                sea = solve();
                if (!sea) {
                    end = true;
                    break;
                }
                solved = sea[0];
            }

            // get the next coordinate to move on,, in the solved path array
            var popp = solved.shift();
            x = popp[0];
            y = popp[1];

            // ------------ Used for collecting data for model training -----------
            // collectNNDataSeeingAsSnake(snake, target);

            // Update the frame with a delay of "speed"
            totalsteps += 1;
            await sleep(speed);
            i += 1;
        }

        if (exit) break;

        if (end)
            // End of game
            break;

        // Else the snake has eaten the apple
        tempDirection = getAiDirection(x, y, snake[0][0], snake[0][1]);
        snake = [[x, y], ...snake];

        changeColorAt(x, y, red);
        changeColorAt(snake[1][0], snake[1][1], white);

        // For Collecting data for training Neural Network Model
        // var t = snake.pop();
        // changeColorAt(t[0], t[1], black);
        // collectNNDataSeeingAsSnake(snake, target);

        generateTarget(snake);
        setScore();
        totalsteps += 1;
        // Generate a new target(apple) after eating the apple
    }
}

function solve() {
    // AI Solver

    console.log("solving");
    // game = self.game
    var temptarget = target;
    var typeSearch = "A*";
    // Can also be "BFS/Dijxtra's" and "DFS", for other search algo
    var TakelongPath = false;
    // To decide whether to take long or short path

    function newStackSearch() {
        // Sets a 2d array and a stack for solving purpose
        var stack = [];
        var searched = [];
        for (let i = 0; i < row; i++) {
            var arrr = [];
            for (let j = 0; j < col; j++) {
                if (getColor(i, j) === white) arrr.push(true);
                else arrr.push(false);
            }
            searched.push(arrr);
        }

        searched[snake[0][0]][snake[0][1]] = true;
        var x = snake[0][0];
        var y = snake[0][1];
        stack.push([x, y, 0]);
        // push is same as push for stack

        return [stack, searched];
    }

    function search(maxsearches = 100000, targ = null) {
        // Searches a path from head to the destination and returns it as a array of coordinates
        var x = snake[0][0];
        var y = snake[0][1];

        // maxse = maxsearches, playing with this can improve AI performance
        var maxse = 1000000;
        var t = null;
        var i = 1;

        while ((x !== temptarget[0] || y !== temptarget[1]) && i < maxse) {
            if (typeSearch === "DFS") {
                if (stack.length === 0) {
                    break;
                }
                t = stack.pop();
            } else if (typeSearch === "BFS/Dijxtra's") {
                if (stack.length === 0) {
                    break;
                }
                t = stack.shift();
            } else {
                if (stack.length === 0) {
                    // Debugger, Not of use
                    if (targ !== null) console.log("Returning false");
                    return false;
                }

                // t = stack.pop(0);
                t = stack.shift();
            }

            x = t[0];
            y = t[1];
            pushNeighboursOfT(t);
            // Push neighbours of x,y for A* searching

            // Sorting the elements in stack according tp the heuristic defined in sorting function
            if (typeSearch === "A*") stack.sort(sorting);
            i += 1;
        }

        // After searching get the path from head to destination
        var a = [];
        if (x === temptarget[0] && y === temptarget[1]) {
            a = getPath(temptarget);
            return [a, true];
        } else {
            a = getPath(stack.shift());
            return [a, false];
        }
    }

    function sorting(a, b) {
        // A* sorting
        var a_distance = a[2];
        var a_hueristic =
            Math.abs(temptarget[0] - a[0]) + Math.abs(temptarget[1] - a[1]);
        var b_distance = b[2];
        var b_hueristic =
            Math.abs(temptarget[0] - b[0]) + Math.abs(temptarget[1] - b[1]);
        if (TakelongPath)
            // For reverse A* (long path), in case the snake has stuck in loop
            return (
                3 * b_distance + b_hueristic - (3 * a_distance + a_hueristic)
            );
        // Simple A*
        else
            return (
                a_distance + 3 * a_hueristic - (b_distance + 3 * b_hueristic)
            );
    }

    function pushNeighboursOfT(t) {
        // Push neighbours of x,y for path searching purpose
        var x = t[0];
        var y = t[1];
        if (x < row - 1 && searched[x + 1][y] === false) {
            searched[x + 1][y] = [x, y];
            stack.push([x + 1, y, t[2] + 1]);
        }
        if (x > 0 && searched[x - 1][y] === false) {
            searched[x - 1][y] = [x, y];
            stack.push([x - 1, y, t[2] + 1]);
        }
        if (y > 0 && searched[x][y - 1] === false) {
            searched[x][y - 1] = [x, y];
            stack.push([x, y - 1, t[2] + 1]);
        }
        if (y < col - 1 && searched[x][y + 1] === false) {
            searched[x][y + 1] = [x, y];
            stack.push([x, y + 1, t[2] + 1]);
        }
    }

    function getPath(fromtemptarget) {
        // helper for search
        // After searching get the Path and return at as a array
        try {
            var x = searched[fromtemptarget[0]][fromtemptarget[1]][0];
            var y = searched[fromtemptarget[0]][fromtemptarget[1]][1];
            var arr = [[fromtemptarget[0], fromtemptarget[1]]];
            var a = 0;
            var b = 0;

            while (x !== snake[0][0] || y !== snake[0][1]) {
                arr.push([x, y]);
                a = searched[x][y][0];
                b = searched[x][y][1];
                x = a;
                y = b;
            }

            return arr.reverse();
        } catch (e) {
            return [temptarget];
        }
    }

    function getEscapetemptarget() {
        // Used when the snake stucks in a loop, (loop made by the snake itself)

        // The idea is to go to that point, where this loop will open (it will open to the point which is nearest to the tail and inside the loop)

        // And if we take longest path to this point, the loop will have sufficient time to open.

        // This function returns this required point.
        var woww = newStackSearch();
        var sta = woww[0];
        var searc = woww[1];
        // Create a new 2d Array and stack for searching purpose
        var ggBlack = getValueNeighbour(snake[0][0], snake[0][1], searc, false);
        // gets a neighbour of the head

        if (ggBlack === null)
            // AI has failed and game will end because the head is blocked
            return null;

        // Recognizing the area of loop by floodfill algo
        floodfill(ggBlack[0], ggBlack[1], searc);

        // Finding the point nearest to the tail and inside the loop region
        var p = 0;
        var q = 0;
        var wow = 0;
        // Just some trick
        for (let x = 0; x < snake.length; x++) {
            p = snake[snake.length - x - 1][0];
            q = snake[snake.length - x - 1][1];
            if (
                (p < row - 1 && searc[p + 1][q] === 5) ||
                (p > 0 && searc[p - 1][q] === 5) ||
                (q < col - 1 && searc[p][q + 1] === 5) ||
                (q > 0 && searc[p][q - 1] === 5)
            ) {
                if (wow === 1) break;
                wow = 1;
            }
        }
        var bneigh = getValueNeighbour(p, q, searc, 5);

        return bneigh !== null ? bneigh : null;
    }

    function floodfill(a, b, searc) {
        // Floodfill to recognize the loop region
        var pstack = [[a, b]];
        while (pstack.length !== 0) {
            var t = pstack.pop();
            var x = t[0];
            var y = t[1];
            if (
                x >= 0 &&
                x < row &&
                y < col &&
                y >= 0 &&
                searc[x][y] !== true &&
                searc[x][y] !== 5
            ) {
                // self.game[x][y].configure(bg='#faf884')
                pstack.push([x + 1, y]);
                pstack.push([x - 1, y]);
                pstack.push([x, y + 1]);
                pstack.push([x, y - 1]);
                searc[x][y] = 5;
            }
        }
    }

    function getValueNeighbour(p, q, searc, value = false) {
        if (p < row - 1 && searc[p + 1][q] === value) return [p + 1, q];
        if (p > 0 && searc[p - 1][q] === value) return [p - 1, q];
        if (q < col - 1 && searc[p][q + 1] === value) return [p, q + 1];
        if (1 > 0 && searc[p][q - 1] === value) return [p, q - 1];
        return null;
    }

    // flag = false
    var woww = newStackSearch();
    var stack = woww[0];
    var searched = woww[1];
    // Creates a new 2d array and stack for searching purpose

    // This is used to search for a maximum distance, can improve performance by changing the function/coefficient

    // Here this depend on the snake length linearly
    var coeff = 3;
    var maxsearches = 30 + snake.length * coeff;

    // For long Snake (> 200), we prefer to take long path to avoid loops
    if (snake.length > 200 && snake.length % 10 === 0) TakelongPath = true;

    var sear = search(maxsearches);

    // search the path, By default we search the shortest path (A*)

    // if sear === false, this means we cannot get the path from the head to the apple directly

    // so we need to go out of the loop, (loop made by the snake itself)

    // The idea is to go to that point, where this loop will open (it will open to the point which is nearest to the tail and inside the loop)

    // And if we take longest path to this point, the loop will have sufficient time to open.
    if (sear === false) {
        var escapetemptarget = getEscapetemptarget();
        // Returns the required escape point as described above
        woww = newStackSearch();
        stack = woww[0];
        searched = woww[1];
        // Creates a new 2d array and stack for searching purpose
        if (escapetemptarget !== null) {
            TakelongPath = true;
            // Flag for taking longPath
            temptarget = escapetemptarget;
            var targ = escapetemptarget;
            sear = search(1000000, targ);
        }
        // search the path
        // AI has failed as the head has no path to go.
        else return false;
    }
    var searchedPath = sear[0];
    // sear[0] is the array of path coordinates
    if (sear[1])
        // Path is to go to the temptarget
        return [searchedPath, true];
    // Path is to escape the loop or going towards the temptarget
    else return [searchedPath, false];
}

function generateTarget(snake) {
    var x = Math.round((Math.random() * 1000) % (row - 1));
    var y = Math.round((Math.random() * 1000) % (col - 1));

    while (containArray(snake, [x, y])) {
        x = Math.round((Math.random() * 1000) % (row - 1));
        y = Math.round((Math.random() * 1000) % (col - 1));
    }
    target[0] = x;
    target[1] = y;
    changeColorAt(x, y, yellow);
}

//--------------------------------------------------------------------------------------------------------
// Manual Play

async function playManual() {
    while (true) {
        if (pause === true) {
            console.log("sleep");
            if (exit) break;
            await dopause();
            continue;
        }
        if (exit) break;
        var newHead = await getNewHead(snake);
        if (newHead === null) {
            // console.log("No Head Found");
            break;
        }
        snake = [[newHead[0], newHead[1]], ...snake];
        changeColorAt(newHead[0], newHead[1], red);
        changeColorAt(snake[1][0], snake[1][1], white);
        if (newHead[2] !== true) {
            var t = snake.pop();
            changeColorAt(t[0], t[1], black);
        } else {
            generateTarget(snake);
            setScore();
            // console.log("New Target", target);
        }

        await sleep(speed);
    }
}

async function getNewHead(snake, direc = 5, targ = 5) {
    // direc and targ is optional parameter with default value of 5. Given only for genetic mode. 5 means nothing here.

    // 0 Up, 1 Down, 2 Left, 3 Right
    var NN = false;
    if (ai == 2) NN = true; // Setting this true will activate neural network mode in manual mode
    var d = null;

    if (direc != 5) {
        d = direc; // For genetic evolution
    } else {
        if (NN) {
            d = await predict(); // For Neural Net Training
        } else {
            d = direction; // For Manual
        }
    }

    tempDirection = d;

    var head = snake[0];
    var x = head[0];
    var y = head[1];
    var a = x;
    var b = y;
    if (d === 0) {
        if (x !== 0) a -= 1;
        else {
            a = row - 1;
            if (ai === 2) return null;
        }
    } else if (d === 1) {
        if (x !== row - 1) a += 1;
        else {
            a = 0;
            if (ai === 2) return null;
        }
    } else if (d === 2) {
        if (y !== 0) b -= 1;
        else {
            b = col - 1;
            if (ai === 2) return null;
        }
    } else {
        if (y !== col - 1) b += 1;
        else {
            b = 0;
            if (ai === 2) return null;
        }
    }

    if (targ != 5) {
        if (a === targ[0] && b === targ[1]) {
            // console.log("Ate Apple");
            return [a, b, true];
        }
    } else {
        if (a === target[0] && b === target[1]) {
            // console.log("Ate Apple");
            return [a, b, true];
        }
    }

    if (containArray(snake, [a, b])) {
        if (direc != 5) {
            return await getNewHead(
                snake,
                getAiDirection(
                    snake[0][0],
                    snake[0][1],
                    snake[1][0],
                    snake[1][1]
                ),
                targ
            );
        } else {
            console.log(
                "head",
                x,
                y,
                "line 170: Returning null for no head found, (a,b):",
                a,
                b,
                "color:",
                getColor(a, b)
            );
            return null;
        }
    }

    return [a, b, false];
}

function setScore() {
    document.getElementById("score").innerHTML =
        "Score: " + (snake.length - 5).toString();
}

async function dopause() {
    clearInterval(timer);
    while (pause) {
        await sleep(200);
    }
    if (!exit) setClock();
}

function sleep(time) {
    return new Promise((resolve) => setTimeout(resolve, time));
}

function containArray(arr, x) {
    // Checks if the array arr has x in it
    var comp = JSON.stringify(x);
    var found = false;
    arr.map((i) => {
        if (comp === JSON.stringify(i)) {
            found = true;
        }
    });
    return found;
}

function getAiDirection(x, y, a, b) {
    // 0 Up, 1 Down, 2 Left, 3 Right
    // a,b is head // x,y is going to
    var d = null;
    if (x == a) {
        if (y > b) {
            d = 3;
        } else {
            d = 2;
        }
    } else if (y == b) {
        if (x > a) {
            d = 1;
        } else {
            d = 0;
        }
    }
    return d;
}

// Controler for Manual and Logical AI play
async function main(OldSnake) {
    // Controls Manual and Logical AI mode
    if (ai !== 5) {
        while (true) {
            await setup(OldSnake);
            if (ai) {
                console.log("PlayingAI");
                await playAI();
            } else {
                console.log("PlayingManual");
                await playManual();
            }
            if (exit) {
                break;
            }
            OldSnake = null;
            await sleep(1000);
        }
    }
    ended = true;
}

async function setup(OldSnake = null, Oldtime = null) {
    totalDataDirection = [];
    ctx.fillStyle = "black";
    await ctx.fillRect(0, 0, canvas.width, canvas.height);
    // showGrid();
    snake =
        ai != true
            ? [
                  [20, 45],
                  [20, 44],
                  [20, 43],
                  [20, 42],
                  [20, 41],
              ]
            : [
                  [20, 45],
                  [20, 44],
                  [20, 43],
                  [20, 42],
                  [20, 41],
                  [20, 40],
                  [20, 39],
                  [20, 38],
                  [20, 37],
                  [20, 36],
                  [20, 35],
                  [20, 34],
                  [20, 33],
                  [20, 32],
                  [20, 31],
                  [20, 30],
                  [20, 29],
                  [20, 28],
                  [20, 27],
                  [20, 26],
              ];
    if (OldSnake !== null) {
        snake = OldSnake;
    }
    speed = ai ? 0 : 40;
    direction =
        snake[0][0] === snake[1][0]
            ? snake[0][1] - snake[1][1] > 0
                ? 3
                : 2
            : snake[0][0] - snake[1][0] > 0
            ? 1
            : 0;
    console.log("direction", direction);
    snake.forEach((coor) => {
        changeColorAt(coor[0], coor[1], white);
    });
    changeColorAt(snake[0][0], snake[0][1], red);

    if (OldSnake === null) {
        generateTarget(snake);
    } else {
        changeColorAt(target[0], target[1], yellow);
    }
    setScore();
    if (timer !== null && OldSnake === null) clearInterval(timer);
    if (ai !== 5 && OldSnake === null) setClock();

    // black - 0, white - 1, yellow - 2, red - 3
    totalsteps = 0;
    await sleep(100);
}

setup();

// ---------------------------------------------------------------------
// Only used to send data to my server for training purpose, Not used Here
// import { sendData, getData } from "./writeData.js";
// async function sendGridDataToServer() {
//     var data = {
//         game: totalSnakeSmartNNData,
//         direction: totalDataDirection,
//     };
//     console.log(
//         "1st data, 200 data",
//         totalSnakeSmartNNData[0],
//         "\n",
//         totalSnakeSmartNNData[200],
//         "\n direction: 0: ",
//         totalDataDirection[0],
//         "1: ",
//         totalDataDirection[200]
//     );
//     await sendData(data, "/snakeGame/");
// }

// async function predict() {
//     var predictionInput = collectNNDataSeeingAsSnake(snake, target);

//     var data = {
//         state: predictionInput,
//     };
//     var predictedDire = null;
//     await sendData(data, "/predict").then(
//         (res) => (predictedDire = JSON.parse(res).direction)
//     );
//     console.log("predictedDirec", predictedDire);
//     return predictedDire;
// }

// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Event Handlers

document.addEventListener("keydown", async (event) => {
    console.log(`key=${event.key},code=${event.KeyCode}`);
    console.log(event.keyCode);
    // 0 Up, 1 Down, 2 Left, 3 Right
    if (event.keyCode === 87) {
        speed -= 5;
    } else if (event.keyCode === 83) {
        speed += 5;
    } else if (event.keyCode === 32) {
        pause = !pause;
        var hid = document.querySelector(".bg-modal").style.display;
        document.querySelector(".bg-modal").style.display =
            hid === "flex" ? "none" : "flex";
    } else if (event.keyCode === 37) {
        if (direction !== 3) direction = 2;
    } else if (event.keyCode === 38) {
        if (direction !== 1) direction = 0;
    } else if (event.keyCode === 39) {
        if (direction !== 2) direction = 3;
    } else if (event.keyCode === 40) {
        if (direction !== 0) direction = 1;
    } else if (event.keyCode === 17) {
        // COntrol Key
        shiftPlay();
    } else if (event.keyCode === 8) {
        // Backspace
        // This is for sending data to server for training model, not used here.
        // pause = !pause;
        // var hid = document.querySelector(".bg-modal").style.display;
        // document.querySelector(".bg-modal").style.display =
        //     hid === "flex" ? "none" : "flex";
        // await sendGridDataToServer();
    }
});

document.getElementById("playAI").addEventListener("click", async function () {
    document.querySelector(".bg-modal").style.display = "none";
    setNormalDisplay();
    clearInterval(timer);
    if (pause) {
        exit = true;
        await sleep(500);
        pause = false;
        // debugger;
    }
    await sleep(1000);
    exit = false;
    ai = true;

    main();
});

document.getElementById("Manual").addEventListener("click", async function () {
    document.querySelector(".bg-modal").style.display = "none";
    setNormalDisplay();
    clearInterval(timer);
    if (pause) {
        exit = true;
        await sleep(500);
        pause = false;
        // debugger;
    }
    await sleep(1000);
    exit = false;
    ai = false;
    main();
});

// Handling Sliders
var mutationRate = 5;
var mutationChange = 20;
var sliderMutationRate = document.getElementById("rate");
var sliderMutationChange = document.getElementById("change");
var outputRate = document.getElementById("rateValue");
var outputChange = document.getElementById("changeValue");
outputRate.innerHTML = "&nbsp" + sliderMutationRate.value + " %";
outputChange.innerHTML = "&nbsp" + sliderMutationChange.value + " %";
sliderMutationRate.oninput = function () {
    outputRate.innerHTML = "&nbsp" + this.value + " %";
    mutationRate = this.value;
};
sliderMutationChange.oninput = function () {
    outputChange.innerHTML = "&nbsp" + this.value + " %";
    mutationChange = this.value;
};

function setGeneticDiplay() {
    document.getElementById("headerlegend").src = "./assets/space.png";
    document.getElementById("modal-content").style.display = "none";
    document.getElementById("geneticModal").style.display = "flex";
    document.getElementById("slidecontainer").style.display = "flex";
    document.getElementById("timer").style.display = "none";
}

function setNormalDisplay() {
    document.getElementById("headerlegend").src = "./assets/headerlegend.png";
    document.getElementById("generation").style.display = "none";
    document.getElementById("score").style.display = "initial";
    document.getElementById("slidecontainer").style.display = "none";
    document.getElementById("timer").style.display = "flex";
}

document.getElementById("genetic").addEventListener("click", async function () {
    setGeneticDiplay();
    document.getElementById("continue").onclick = async () => {
        document.getElementById("geneticModal").style.display = "none";
        document.getElementById("modal-content").style.display = "flex";
        document.querySelector(".bg-modal").style.display = "none";
        clearInterval(timer);
        if (pause) {
            exit = true;
            await sleep(500);
            pause = false;
        }
        await sleep(2000);
        exit = false;
        ai = 2;
        var species = await startGeneticEvolution();
    };
});

async function shiftPlay() {
    var Oldsnake = snake;
    exit = true;
    while (!ended) {
        await sleep(100);
    }
    ended = false;
    ai = !ai;
    exit = false;
    main(Oldsnake);
}

async function setClock(oldtime = 0) {
    var time = oldtime;
    timer = setInterval(() => {
        var min = Math.floor(time / 60);
        var sec = Math.floor(time % 60);
        time += 1;

        document.getElementById("timer").innerHTML =
            "Time: " + min + ":" + sec.toString();
    }, 1000);
}

function changeColorAt(x, y, color) {
    // 2 - Head, 1 - Body, 0 - Black
    ctx.beginPath();
    // ctx.clearRect(wid * y, hei * x, wid, hei);
    ctx.fillStyle = color;
    ctx.fillRect(wid * y, hei * x, wid, hei);
    // ctx.strokeRect(wid * y, hei * x, wid, hei);
}

function rgbToHex(r, g, b) {
    if (r > 255 || g > 255 || b > 255) throw "Invalid color component";
    return ((r << 16) | (g << 8) | b).toString(16);
}

function getColor(x, y) {
    var p = ctx.getImageData(y * wid + 5, x * hei + 5, 1, 1).data;
    var hex = "#" + ("000000" + rgbToHex(p[0], p[1], p[2])).slice(-6);
    return hex;
}
