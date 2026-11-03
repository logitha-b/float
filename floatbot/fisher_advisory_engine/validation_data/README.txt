Place INCOIS PFZ advisory CSV files here for real-data validation.

File naming convention: incois_pfz_YYYY-MM-DD.csv

Required columns per file:
  date         : ISO date (YYYY-MM-DD)
  lat          : float (decimal degrees)
  lon          : float (decimal degrees)
  pfz_active   : 1 (fishing zone active) or 0
  district     : Tamil Nadu district name

Optional columns:
  front_noted  : 1 or 0
  chl_category : HIGH / MODERATE / LOW

Source: https://incois.gov.in/portal/osf/pfz.jsp

Do NOT commit real PFZ data to version control without verifying the
licence terms from INCOIS.
