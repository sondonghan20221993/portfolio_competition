module.exports = {
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL || 'https://dwiwijaya.com',
  generateRobotsTxt: true,
  alternateRefs: [
    {
      href: 'https://dwiwijaya.com/en',
      hreflang: 'en',
    },
    {
      href: 'https://dwiwijaya.com',
      hreflang: 'id',
    },
  ],
};
