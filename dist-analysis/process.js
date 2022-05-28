import axios from "axios";
import fs from "fs";
import proj4 from "proj4"

proj4.defs([
    [
        "EPSG:2097",
        "+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs +towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43"
    ],
]);

async function parseJSON(path, converter) {
    process.stdout.write(`Parsing (${path})... \r`);
    const file = fs.readFileSync(path).toString();
    const original = JSON.parse(file)["DATA"];

    const data = [];
    let iterCount = 0;
    for (const record of original) {
        process.stdout.write(`Parsing (${path})... (${((++iterCount)/original.length*100).toFixed(2)} %)\r`);
        const res = await converter(record);
        if (res) {
            data.push(res);
        }
    }
    process.stdout.write(`\n`);
    return data;
}

async function getCoord(addr) {
    if (!getCoord.cache) {
        getCoord.cache = {};
    } else if (addr in getCoord.cache) {
        return getCoord.cache[addr];
    }
    if (Object.keys(getCoord.cache).length % 100 === 0) {
        fs.writeFileSync("coordCache.json", JSON.stringify(getCoord.cache, null, 4));
    }

    const clientID = "ifl298e7ts";
    const clientSecret = "dHhufojNXgL2IQXhATdqkQ9pGljYofBCmR1YdSZt";
    const encoded = encodeURIComponent(addr);
    const url = `https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query=${encoded}`;

    try {
        const res = await axios.get(url, { 
            params: addr,
            headers: {
                "X-NCP-APIGW-API-KEY-ID": clientID,
                "X-NCP-APIGW-API-KEY": clientSecret
            }
        });
    
        const { x, y } = res.data.addresses[0];
        getCoord.cache[addr] = [ Number(x), Number(y) ];
        return getCoord.cache[addr];
    } catch {
        getCoord.cache[addr] = null;
        return getCoord.cache[addr];
    }
}

// 위/경도
const EPSG4326 = (kind, xCol, yCol) => async (record) => {
    if (record[xCol] && record[yCol]) {
        return {
            datatype: kind,
            x: Number(record[xCol]),
            y: Number(record[yCol])
        };
    }
    return null;
};

// 중부 원점
const EPSG2097 = (kind, xCol, yCol) => async (record) => {
    if (record[xCol] && record[yCol]) {
        const [lo, la] = proj4("EPSG:2097", "EPSG:4326", [Number(record[xCol]), Number(record[yCol])]);
        return {
            datatype: kind,
            x: lo,
            y: la
        };
    }
    return null;
};

const Address = (kind, addrCol) => async (record) => {
    if (record[addrCol]) {
        const coord = await getCoord(record[addrCol]);
        if (coord === null) {
            return null;
        }
        return {
            datatype: kind,
            x: coord[0],
            y: coord[1]
        };
    }
    return null;
};

async function main() {
    if (fs.existsSync("data/coordlist.json")) {
        getCoord.cache = JSON.parse(fs.readFileSync("./data/coordlist.json").toString())
    }
    const result = [
        ...await parseJSON("data/교육 시설/서울특별시 어린이집 정보(표준 데이터).json", EPSG4326("nursery", "lo", "la")),
        ...await parseJSON("data/교육 시설/서울특별시 유치원 일반현황.json", Address("kindergarten", "addr")),
        ...await parseJSON("data/교육 시설/서울특별시 학교 기본정보.json", Address("school", "org_rdnma")),
        ...await parseJSON("data/교육 시설/서울특별시 학원 교습소정보.json", Address("academy", "fa_rdnma")),
    
        ...await parseJSON("data/불편 시설/서울특별시 단란주점영업 인허가 정보.json", EPSG2097("karaoke-bar", "x", "y")),
        ...await parseJSON("data/불편 시설/서울특별시 유흥주점영업 인허가 정보.json", EPSG2097("entertainment-bar", "x", "y")),
    
        ...await parseJSON("data/편의 시설/서울특별시 대규모점포 인허가 정보.json", EPSG2097("super-super-market", "x", "y")),
        ...await parseJSON("data/편의 시설/서울특별시 병원 인허가 정보.json", EPSG2097("hospital", "x", "y")),
        ...await parseJSON("data/편의 시설/서울특별시 일반음식점 인허가 정보.json", EPSG2097("restaurant", "x", "y")),
        ...await parseJSON("data/한강 경로.json", EPSG4326("han-river", "lng", "lat"))
    ];
    
    fs.writeFileSync("processed.json", JSON.stringify(result, null, 4));
    fs.writeFileSync("data/coordlist.json", JSON.stringify(getCoord.cache, null, 4));
}

main();