import { ThemeProvider } from "./theme/ThemeContext";
import { I18nProvider } from "./i18n/I18nContext";
import { ConfirmProvider } from "./components/ConfirmDialog";
import { AppShell } from "./components/AppShell";
import { SplashScreen } from "./components/SplashScreen";
import { useAppState } from "./hooks/useAppState";

export function App() {
  const state = useAppState();
  return (
    <ThemeProvider>
      <I18nProvider>
        <ConfirmProvider>
          <AppShell {...state} />
          <SplashScreen />
        </ConfirmProvider>
      </I18nProvider>
    </ThemeProvider>
  );
}
