import fs from "fs"
import csv from "fast-csv"

const proc = JSON.parse(fs.readFileSync("data/coordCache.json").toString())
const converted = {}
for (const key in proc) {
    if (proc[key] === null) {
        converted[key] = null;
    } else {
        converted[key] = [proc[key][1], proc[key][0]];
    }
}

fs.createReadStream("data/_coordlist.csv")
    .pipe(csv.parse({ headers: false }))
    .on("error", error => console.log(error))
    .on("data", row => {
        converted[row[0]] = [Number(row[1]), Number(row[2])];
    })
    .on("end", () => {
        fs.writeFileSync("data/coordlist.json", JSON.stringify(converted, null, 4));
        const csvLines = [];
        for (const key in converted) {
            if (converted[key]) {
                csvLines.push(`\"${key}\",${converted[key][0]},${converted[key][1]}`);
            }
        }
        fs.writeFileSync("data/coordlist.csv", csvLines.join("\n"));
    })