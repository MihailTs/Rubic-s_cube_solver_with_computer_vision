async function sendSolveRequest() {
        try {
            let object = {
                front: sharedCubeState.frontSide,
                top: sharedCubeState.topSide,
                left: sharedCubeState.leftSide,
                right: sharedCubeState.rightSide,
                bottom: sharedCubeState.bottomSide,
                back: sharedCubeState.backSide
            };

            object = Object.fromEntries(
                Object.entries(object).map(([key, element]) => [
                    key,
                    element.map(elem => sharedCubeState.number_to_color[elem])
                ])
            );

            // console.log(object);

            const response = await fetch("http://localhost:8000/solve", {
                method: "POST",
                headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
                },
                body: JSON.stringify(object)
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            console.log("Server response:", data);

            if (data.status == 'ok') {
                // console.log(`Image processed`);

                // for(let i = 0; i < 9; i++) {
                //     // don't change center color
                //     if(i == 4) continue;
                //     side[i] = sharedCubeState.number_to_color[data.predictions[i]];
                // }
            } else {
                console.error("Processing failed:", data.message);
            }
        } catch (error) {
            window.alert("Upload failed:", error);
        }

    }