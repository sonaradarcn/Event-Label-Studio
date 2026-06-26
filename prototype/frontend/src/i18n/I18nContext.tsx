import { createContext, useCallback, useContext, useEffect, useState } from "react";
import enUS from "./locales/en-US.json";
import zhCN from "./locales/zh-CN.json";

type Locale = string;
const STORAGE_KEY = "locale";

const translations: Record<string, Record<string, string>> = {
  "en-US": enUS,
  "zh-CN": zhCN,
};

const I18nContext = createContext<{
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string) => string;
  locales: { code: string; name: string }[];
}>({
  locale: "en-US",
  setLocale: () => {},
  t: (k) => k,
  locales: [],
});

export const availableLocales = [
  { code: "en-US", name: "English" },
  { code: "zh-CN", name: "简体中文" },
];

function resolve(obj: Record<string, string>, key: string): string {
  return obj[key] ?? key;
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    return localStorage.getItem(STORAGE_KEY) || "en-US";
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, locale);
  }, [locale]);

  const setLocale = useCallback((l: Locale) => setLocaleState(l), []);

  const t = useCallback(
    (key: string) => resolve(translations[locale] ?? translations["en-US"], key),
    [locale]
  );

  return (
    <I18nContext.Provider value={{ locale, setLocale, t, locales: availableLocales }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}
