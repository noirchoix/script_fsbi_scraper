const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export type FoodRecord = {
  entity_id: number | null;
  alias: string;
  category: string;
  scientific_name: string;
  molecule_ids: number[];
  source_url: string;
};

export type MoleculeRecord = {
  pubchem_id: number | null;
  common_name: string;
  flavor_profile: string[];
  source_url: string;
};

export type FlavorTerm = {
  term: string;
  count: number;
};

export type ScrapeResult = {
  query: string;
  status: 'ok' | 'no_results' | string;
  summary: {
    food_matches: number;
    molecule_matches: number;
    unique_flavor_terms: number;
  };
  flavor_profile: FlavorTerm[];
  foods: FoodRecord[];
  molecules: MoleculeRecord[];
  trace: string[];
  warnings: string[];
};

export async function scrapeFood(food: string, maxFoods = 3, maxCompounds = 25): Promise<ScrapeResult> {
  const response = await fetch(`${API_BASE_URL}/api/scrape`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      food,
      max_foods: maxFoods,
      max_compounds: maxCompounds
    })
  });

  if (!response.ok) {
    let message = `FSBI API returned ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch {
      // Keep default message.
    }
    throw new Error(message);
  }

  return response.json();
}
