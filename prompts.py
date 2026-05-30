import torch
import torch.nn.functional as F


CLASS_NAMES = {
    "Trento": ["Buildings", "Woods", "Roads", "Apples", "ground", "Vineyard"],
    "Houston": [
        "Healthy grass",
        "Stressed grass",
        "Synthetic grass",
        "Trees",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Highway",
        "Railway",
        "Parking Lot 1",
        "Parking Lot 2",
        "Tennis Court",
        "Running Track",
    ],
    "MUUFL": [
        "Trees",
        "Mostly grass",
        "Mixed ground surface",
        "Dirt and sand",
        "Road",
        "Water",
        "Building shadow",
        "Building",
        "Sidewalk",
        "Yellow curb",
        "Cloth panels",
    ],
}


PROMPT_SETS = {
    "Trento": {
        "name": {
            "Buildings": [
                "land cover class buildings",
                "remote sensing class buildings",
            ],
            "Woods": [
                "land cover class woods",
                "remote sensing class woods",
            ],
            "Roads": [
                "land cover class roads",
                "remote sensing class roads",
            ],
            "Apples": [
                "land cover class apple orchard",
                "remote sensing class apple orchard",
            ],
            "ground": [
                "land cover class bare ground",
                "remote sensing class bare ground",
            ],
            "Vineyard": [
                "land cover class vineyard",
                "remote sensing class vineyard",
            ],
        },
        "spectral": {
            "Buildings": [
                "roof masonry tile concrete artificial non vegetated flat reflectance",
                "concrete tile masonry roof artificial flat non vegetated reflectance",
            ],
            "Woods": [
                "forest dense closed canopy high biomass continuous tree crown NIR plateau",
                "dense closed forest continuous tree crowns high biomass NIR plateau",
            ],
            "Roads": [
                "asphalt road black bitumen pavement low NIR linear impervious surface",
                "black bitumen asphalt road linear pavement low NIR impervious surface",
            ],
            "Apples": [
                "apple orchard deciduous fruit trees separated crowns grass understory",
                "deciduous apple orchard separated fruit tree crowns grass understory",
            ],
            "ground": [
                "bare plowed mineral soil dry earth no canopy no asphalt no roof",
                "dry plowed mineral soil bare earth no canopy no asphalt no roof",
            ],
            "Vineyard": [
                "vineyard grapevine trellis rows sparse foliage exposed interrow soil",
                "grapevine vineyard trellis rows sparse foliage exposed interrow soil",
            ],
        },
        "spectral_lidar": {
            "Buildings": [
                "building roof planar elevated angular walls concrete tile artificial surface",
                "planar elevated building roof angular concrete artificial surface",
            ],
            "Woods": [
                "woodland forest tall rough continuous canopy uneven natural tree crowns",
                "tall rough continuous forest woodland uneven connected tree crowns",
            ],
            "Roads": [
                "road asphalt flat linear paved corridor smooth ground level strip",
                "flat linear asphalt road paved corridor smooth low elevation strip",
            ],
            "Apples": [
                "apple orchard regular grid medium fruit tree crown mounds",
                "regular apple orchard medium rounded fruit tree crown grid",
            ],
            "ground": [
                "bare dirt field flat mineral soil no vertical object diffuse terrain",
                "flat bare mineral dirt field diffuse terrain no vertical object",
            ],
            "Vineyard": [
                "vineyard low trellis parallel rows sparse canopy soil corridors",
                "low trellis vineyard parallel rows sparse canopy exposed soil corridors",
            ],
        },
    }
}


def _name_prompt_set(class_names):
    return {
        class_name: [
            f"land cover class {class_name.lower()}",
            f"remote sensing class {class_name.lower()}",
        ]
        for class_name in class_names
    }


PROMPT_SETS["Houston"] = {
    "name": _name_prompt_set(CLASS_NAMES["Houston"]),
    "spectral": {
        "Healthy grass": [
            "healthy green grass dense vegetation high near infrared reflectance",
            "vigorous grass turf strong chlorophyll absorption high vegetation response",
        ],
        "Stressed grass": [
            "stressed grass reduced chlorophyll weaker near infrared vegetation response",
            "dry or stressed grass lower biomass altered red edge reflectance",
        ],
        "Synthetic grass": [
            "artificial turf uniform plastic grass spectral response non photosynthetic",
            "synthetic grass regular surface green material without natural chlorophyll",
        ],
        "Trees": [
            "tree canopy woody vegetation high biomass strong near infrared reflectance",
            "mature trees dense crown vegetation shadowed canopy spectral mixture",
        ],
        "Soil": [
            "bare soil mineral surface dry earth low vegetation response",
            "exposed soil ground material broad smooth reflectance no canopy",
        ],
        "Water": [
            "water dark surface strong near infrared absorption low reflectance",
            "open water very low NIR response smooth dark spectral signature",
        ],
        "Residential": [
            "residential roofs lawns small buildings mixed impervious vegetation spectra",
            "housing area mixed roof asphalt grass tree spectral materials",
        ],
        "Commercial": [
            "commercial buildings large roofs concrete asphalt impervious mixed spectra",
            "urban commercial blocks broad artificial roof and pavement reflectance",
        ],
        "Road": [
            "asphalt road dark pavement low vegetation response linear impervious material",
            "road surface bitumen concrete flat spectral response non vegetated",
        ],
        "Highway": [
            "wide highway asphalt pavement linear transport corridor low vegetation",
            "multi lane highway dark impervious road surface broad linear signature",
        ],
        "Railway": [
            "railway track ballast metal rails linear corridor sparse vegetation",
            "rail line gravel ballast steel track mixed bright dark materials",
        ],
        "Parking Lot 1": [
            "parking lot asphalt concrete open paved surface painted markings",
            "large flat parking pavement impervious surface low vegetation response",
        ],
        "Parking Lot 2": [
            "parking lot paved asphalt concrete vehicle area bright impervious surface",
            "open parking pavement compact artificial surface with painted lines",
        ],
        "Tennis Court": [
            "tennis court colored artificial sports surface uniform high contrast material",
            "rectangular court synthetic coating flat non vegetated spectral response",
        ],
        "Running Track": [
            "running track red synthetic rubber oval surface uniform artificial material",
            "athletic track curved red rubber coating non vegetated surface",
        ],
    },
    "spectral_lidar": {
        "Healthy grass": [
            "healthy grass low flat vegetation surface dense turf high NIR",
            "short dense grass low elevation smooth canopy strong vegetation response",
        ],
        "Stressed grass": [
            "stressed grass low flat vegetation sparse dry canopy reduced NIR",
            "short dry grass low elevation weak vegetation structure reduced biomass",
        ],
        "Synthetic grass": [
            "synthetic grass low flat artificial turf uniform texture no canopy height",
            "flat artificial turf regular surface green material low elevation",
        ],
        "Trees": [
            "trees tall rough canopy elevated vegetation crown structure high biomass",
            "tall tree crowns rough LiDAR height dense vegetation canopy",
        ],
        "Soil": [
            "bare soil flat exposed ground low elevation no vertical structure",
            "mineral earth flat terrain diffuse roughness no canopy no buildings",
        ],
        "Water": [
            "water flat smooth low elevation dark NIR absorption surface",
            "open water smooth planar surface very low reflectance no height",
        ],
        "Residential": [
            "residential mixed low buildings roofs trees lawns varied height structure",
            "housing blocks roof planes vegetation patches moderate LiDAR variation",
        ],
        "Commercial": [
            "commercial large elevated roof planes broad impervious blocks high structure",
            "large urban buildings planar roofs asphalt surroundings strong height edges",
        ],
        "Road": [
            "road flat linear pavement ground level smooth impervious corridor",
            "low elevation linear asphalt strip smooth surface no canopy",
        ],
        "Highway": [
            "highway wide flat linear pavement corridor smooth low elevation",
            "multi lane transport corridor broad asphalt strip ground level",
        ],
        "Railway": [
            "railway narrow linear corridor ballast rails low rough ground structure",
            "rail track linear gravel metal corridor low elevation sparse objects",
        ],
        "Parking Lot 1": [
            "parking lot broad flat paved surface low elevation rectangular open area",
            "flat asphalt parking area smooth impervious surface painted structure",
        ],
        "Parking Lot 2": [
            "parking lot paved open flat impervious surface low height variation",
            "compact parking pavement smooth terrain artificial rectangular area",
        ],
        "Tennis Court": [
            "tennis court flat rectangular sports surface uniform artificial low elevation",
            "colored court planar synthetic surface fenced low structural pattern",
        ],
        "Running Track": [
            "running track flat oval synthetic rubber surface low elevation",
            "athletic track curved artificial surface uniform low height structure",
        ],
    },
}


PROMPT_SETS["MUUFL"] = {
    "name": _name_prompt_set(CLASS_NAMES["MUUFL"]),
    "spectral": {
        "Trees": [
            "tree canopy dense vegetation high near infrared reflectance woody crowns",
            "tall vegetation canopy chlorophyll absorption strong NIR response",
        ],
        "Mostly grass": [
            "mostly grass herbaceous vegetation low canopy high near infrared reflectance",
            "grass cover green vegetation smooth turf chlorophyll spectral response",
        ],
        "Mixed ground surface": [
            "mixed ground surface soil grass pavement heterogeneous spectral mixture",
            "mixed bare ground vegetation and impervious materials varied reflectance",
        ],
        "Dirt and sand": [
            "dirt sand bare mineral surface dry soil bright smooth reflectance",
            "sandy exposed ground low vegetation mineral spectral response",
        ],
        "Road": [
            "asphalt road dark pavement linear impervious surface low NIR",
            "road bitumen concrete paved non vegetated spectral signature",
        ],
        "Water": [
            "water dark surface strong near infrared absorption very low reflectance",
            "open water smooth dark spectral response no vegetation",
        ],
        "Building shadow": [
            "building shadow dark low illumination suppressed reflectance urban surface",
            "shadowed building area low radiance dark spectral signature",
        ],
        "Building": [
            "building roof concrete metal artificial material flat urban reflectance",
            "roof wall impervious built surface non vegetated spectral response",
        ],
        "Sidewalk": [
            "sidewalk concrete pavement bright linear pedestrian surface low vegetation",
            "concrete sidewalk flat impervious material smooth urban reflectance",
        ],
        "Yellow curb": [
            "yellow curb painted concrete narrow bright artificial urban material",
            "painted curb yellow coating small linear impervious spectral response",
        ],
        "Cloth panels": [
            "cloth panels colored fabric artificial material distinctive reflectance",
            "fabric calibration panels uniform man made surface strong color response",
        ],
    },
    "spectral_lidar": {
        "Trees": [
            "trees tall rough canopy elevated vegetation crowns high NIR",
            "tree canopy high LiDAR height irregular crown structure vegetation",
        ],
        "Mostly grass": [
            "mostly grass low flat vegetation smooth turf high NIR",
            "short grass low elevation herbaceous cover smooth canopy",
        ],
        "Mixed ground surface": [
            "mixed ground surface low irregular terrain grass soil pavement mixture",
            "heterogeneous ground low height variation mixed spectral materials",
        ],
        "Dirt and sand": [
            "dirt sand flat bare terrain mineral soil low elevation",
            "exposed sandy ground smooth low height no vegetation canopy",
        ],
        "Road": [
            "road flat linear asphalt pavement low elevation smooth corridor",
            "linear paved surface ground level impervious low roughness",
        ],
        "Water": [
            "water flat smooth surface low elevation dark NIR absorption",
            "open water planar smooth low return surface no structure",
        ],
        "Building shadow": [
            "building shadow adjacent to elevated structures dark low reflectance",
            "shadowed urban surface near building edges low radiance height context",
        ],
        "Building": [
            "building elevated planar roof angular structure artificial surface",
            "built structure high LiDAR edges roof plane impervious material",
        ],
        "Sidewalk": [
            "sidewalk flat narrow concrete path low elevation urban corridor",
            "pedestrian pavement smooth linear ground level artificial surface",
        ],
        "Yellow curb": [
            "yellow curb narrow raised painted edge linear small urban structure",
            "painted curb low raised linear boundary bright artificial material",
        ],
        "Cloth panels": [
            "cloth panels flat small artificial fabric targets uniform surface",
            "colored fabric panels low elevation uniform calibration material",
        ],
    },
}


def get_class_names(dataset):
    if dataset not in CLASS_NAMES:
        raise ValueError(f"No class names are configured for dataset '{dataset}'.")
    return CLASS_NAMES[dataset]


def get_prompt_set(dataset, prompt_mode):
    if prompt_mode is None:
        return None
    if dataset not in PROMPT_SETS:
        raise ValueError(f"No prompt sets are configured for dataset '{dataset}'.")
    if prompt_mode not in PROMPT_SETS[dataset]:
        available = ", ".join(sorted(PROMPT_SETS[dataset]))
        raise ValueError(
            f"Prompt mode '{prompt_mode}' is not available for {dataset}. "
            f"Available modes: {available}."
        )
    return PROMPT_SETS[dataset][prompt_mode]


@torch.no_grad()
def build_text_prototypes(clip_model, tokenizer, dataset, prompt_mode, device):
    class_names = get_class_names(dataset)
    prompt_set = get_prompt_set(dataset, prompt_mode)
    prototypes = []

    for class_name in class_names:
        prompts = prompt_set[class_name]
        tokens = tokenizer(prompts).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        prototype = F.normalize(text_features.mean(dim=0), dim=-1)
        prototypes.append(prototype)

    return torch.stack(prototypes, dim=0)


def prototype_similarity_stats(text_prototypes):
    sim = (text_prototypes @ text_prototypes.T).detach().cpu()
    upper = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
    off_diag = sim[upper[0], upper[1]]
    return {
        "mean_offdiag": float(off_diag.mean().item()),
        "max_offdiag": float(off_diag.max().item()),
        "matrix": sim.numpy(),
    }
